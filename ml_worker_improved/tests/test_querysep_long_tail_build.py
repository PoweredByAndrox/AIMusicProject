from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from worker import querysep_long_tail_build


def _write_bytes(path: Path, size: int = 2048) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)


def _write_audio(path: Path, *, amplitude: float = 0.1, duration_sec: float = 1.0, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_sec * sample_rate))
    audio = np.full((frame_count, 2), float(amplitude), dtype=np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def _write_sine_audio(
    path: Path,
    *,
    amplitude: float = 0.1,
    frequency_hz: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 44100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_sec * sample_rate))
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * float(frequency_hz) * time) * float(amplitude)
    audio = np.stack([mono, mono], axis=1).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def _write_two_tone_audio(
    path: Path,
    *,
    first_amplitude: float,
    first_frequency_hz: float,
    second_amplitude: float,
    second_frequency_hz: float,
    duration_sec: float = 20.0,
    sample_rate: int = 44100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_sec * sample_rate))
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    mono = (
        np.sin(2.0 * np.pi * float(first_frequency_hz) * time) * float(first_amplitude)
        + np.sin(2.0 * np.pi * float(second_frequency_hz) * time) * float(second_amplitude)
    )
    audio = np.stack([mono, mono], axis=1).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def _write_windowed_proxy_leakage_audio(
    path: Path,
    *,
    proxy_frequency_hz: float,
    independent_frequency_hz: float,
    amplitude: float = 0.08,
    leak_start_sec: float = 5.0,
    leak_duration_sec: float = 5.0,
    duration_sec: float = 20.0,
    sample_rate: int = 44100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_sec * sample_rate))
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * float(independent_frequency_hz) * time) * float(amplitude)
    leak_start = int(round(leak_start_sec * sample_rate))
    leak_end = min(frame_count, leak_start + int(round(leak_duration_sec * sample_rate)))
    mono[leak_start:leak_end] = (
        np.sin(2.0 * np.pi * float(proxy_frequency_hz) * time[leak_start:leak_end]) * float(amplitude)
    )
    audio = np.stack([mono, mono], axis=1).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def _write_sparse_active_presence_audio(
    path: Path,
    *,
    active_frequency_hz: float = 660.0,
    active_amplitude: float = 0.08,
    quiet_amplitude: float = 0.00005,
    duration_sec: float = 20.0,
    sample_rate: int = 44100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_sec * sample_rate))
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * float(active_frequency_hz) * time) * float(quiet_amplitude)
    active_end = int(round(10.0 * sample_rate))
    mono[:active_end] = (
        np.sin(2.0 * np.pi * float(active_frequency_hz) * time[:active_end]) * float(active_amplitude)
    )
    audio = np.stack([mono, mono], axis=1).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def _write_source_manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    stem_paths = {
        "drums": tmp_path / "stems" / "drums.flac",
        "bass": tmp_path / "stems" / "bass.flac",
        "other": tmp_path / "stems" / "other.flac",
        "piano": tmp_path / "stems" / "piano.flac",
    }
    for stem_path in stem_paths.values():
        _write_audio(stem_path, amplitude=0.02)

    manifest_path = tmp_path / "stems" / "stem_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stems": {
                    stem_name: {"path": str(stem_path)}
                    for stem_name, stem_path in stem_paths.items()
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path, stem_paths


def _write_dense_source_manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    manifest_path, stem_paths = _write_source_manifest(tmp_path)
    stem_paths.update(
        {
            "vocals": tmp_path / "stems" / "vocals.flac",
            "guitar": tmp_path / "stems" / "guitar.flac",
            "melody": tmp_path / "stems" / "melody.flac",
        }
    )
    frequencies = {
        "drums": 110.0,
        "bass": 165.0,
        "piano": 330.0,
        "vocals": 550.0,
        "guitar": 770.0,
        "melody": 990.0,
    }
    for stem_name, frequency_hz in frequencies.items():
        _write_sine_audio(stem_paths[stem_name], amplitude=0.006, frequency_hz=frequency_hz, duration_sec=20.0)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stems"] = {
        stem_name: {"path": str(stem_path)}
        for stem_name, stem_path in sorted(stem_paths.items())
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, stem_paths


def test_build_querysep_long_tail_stems_requires_banquet_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio" / "track.flac"
    output_dir = tmp_path / "long_tail"
    expected_querysep_output = (
        "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
    )
    expected_local_output = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_audio(audio_path, amplitude=0.25)
    _write_audio(expected_local_output, amplitude=0.08)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    captured_requests: list[dict] = []

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        captured_requests.append({"url": url, "payload": payload, "timeout_seconds": timeout_seconds})
        return {
            "service": "querysep-worker",
            "status": "completed",
            "task": "query_separation",
            "target_stem": "strings_orchestra",
            "named_outputs": {"strings_orchestra": expected_querysep_output},
            "metadata": {"routing_mode": "banquet_byoq"},
            "notes": "Real Banquet BYOQ separation completed.",
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[
            {"local_prefix": str(tmp_path / "audio"), "querysep_prefix": "/querysep/audio"},
            {"local_prefix": str(tmp_path / "stems"), "querysep_prefix": "/querysep/stems"},
            {"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"},
        ],
        request_timeout_seconds=33,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    request_payload = captured_requests[0]["payload"]

    assert result["fallback_used"] is False
    assert result["backend"] == "banquet_byoq"
    assert manifest["fallback_used"] is False
    assert manifest["stems"]["strings_orchestra"]["local_output_path"] == str(expected_local_output)
    assert manifest["stems"]["strings_orchestra"]["status"] == "accepted"
    assert manifest["stems"]["strings_orchestra"]["router"]["status"] == "accepted"
    assert manifest["stems"]["strings_orchestra"]["selected_candidate_id"] == "banquet_byoq_primary"
    assert manifest["stems"]["strings_orchestra"]["selected_output_origin"] == "banquet_byoq"
    assert manifest["stems"]["strings_orchestra"]["selected_output_path"] == str(expected_local_output)
    assert manifest["stems"]["strings_orchestra"]["primary_querysep_output_path"] == expected_querysep_output
    assert manifest["stems"]["strings_orchestra"]["querysep_output_path"] == expected_querysep_output
    assert manifest["stems"]["strings_orchestra"]["quality_gate"]["status"] == "pass"
    assert manifest["quality_gate"]["objective_metrics"]["sdr"] is None
    assert captured_requests[0]["url"] == "http://querysep-worker:8000/run"
    assert captured_requests[0]["timeout_seconds"] == 33
    assert request_payload["raw_step"]["backend"] == "banquet_byoq"
    assert request_payload["dry_run"] is False
    assert request_payload["input_path"] == "/querysep/audio/track.flac"
    assert request_payload["output_root"] == "/querysep/out"
    assert request_payload["build_from_paths"]["drums"] == "/querysep/stems/drums.flac"


def test_build_querysep_long_tail_stems_rejects_proxy_backend(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.flac"
    _write_audio(audio_path)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    with pytest.raises(ValueError, match="banquet_byoq"):
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=tmp_path / "out",
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            backend="clap_router",
        )


def test_build_querysep_long_tail_stems_accepts_string_false_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_audio(audio_path, amplitude=0.25)
    _write_audio(output_path, amplitude=0.08)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        dry_run="false",
    )

    assert result["status"] == "completed"


def test_build_querysep_long_tail_stems_treats_empty_list_build_from_paths_as_manifest_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio" / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_audio(audio_path, amplitude=0.25)
    _write_audio(output_path, amplitude=0.08)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    captured_payloads: list[dict] = []

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        captured_payloads.append(payload)
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        build_from_paths=[],
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[
            {"local_prefix": str(tmp_path / "stems"), "querysep_prefix": "/querysep/stems"},
            {"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"},
        ],
    )

    assert result["status"] == "completed"
    assert captured_payloads[0]["build_from_paths"]["drums"] == "/querysep/stems/drums.flac"


@pytest.mark.parametrize("dry_run_value", [True, "true", "1", "yes", "invalid"])
def test_build_querysep_long_tail_stems_rejects_truthy_or_invalid_dry_run(
    tmp_path: Path,
    dry_run_value: object,
) -> None:
    audio_path = tmp_path / "track.flac"
    _write_audio(audio_path)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    with pytest.raises(ValueError, match="dry_run"):
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=tmp_path / "out",
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            dry_run=dry_run_value,
        )


def test_build_querysep_long_tail_stems_rejects_non_banquet_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_audio(audio_path)
    _write_audio(output_path)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "clap_router"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError, match="non-Banquet"):
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        )


def test_build_querysep_long_tail_stems_fails_when_output_is_not_shared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    _write_audio(audio_path)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(FileNotFoundError, match="not visible"):
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=tmp_path / "out",
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(tmp_path / "out"), "querysep_prefix": "/querysep/out"}],
        )


def test_build_querysep_long_tail_stems_fails_for_too_small_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_audio(audio_path)
    _write_bytes(output_path, size=8)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError, match="too small"):
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
            min_output_bytes=1024,
        )


def test_build_querysep_long_tail_stems_rejects_acoustically_weak_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_audio(audio_path, amplitude=0.3)
    _write_audio(output_path, amplitude=0.001)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError) as exc_info:
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        )

    error_text = str(exc_info.value)
    assert "long-tail model router rejected all production candidates" in error_text
    assert "manifest_path=" in error_text
    assert "long_tail_presence_too_weak" in error_text
    assert "long_tail_presence_too_weak_vs_source" in error_text
    manifest = json.loads((output_dir / "querysep_long_tail_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rejected"
    assert manifest["stems"]["strings_orchestra"]["router"]["status"] == "rejected"
    assert manifest["failure"]["reason"] == "long_tail_model_router_rejected_all_candidates"


def test_build_querysep_long_tail_stems_rejects_sparse_active_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.25, frequency_hz=220.0, duration_sec=20.0)
    _write_sparse_active_presence_audio(output_path)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError) as exc_info:
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        )

    assert "long_tail_active_presence_too_sparse" in str(exc_info.value)
    manifest = json.loads((output_dir / "querysep_long_tail_manifest.json").read_text(encoding="utf-8"))
    active_presence = manifest["stems"]["strings_orchestra"]["quality_gate"]["metrics"]["active_presence"]
    assert active_presence["active_window_count"] == 2
    assert active_presence["required_active_windows"] == 3
    assert manifest["status"] == "rejected"


def test_build_querysep_long_tail_stems_rejects_proxy_copy_correlation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.25, frequency_hz=220.0)
    _write_sine_audio(output_path, amplitude=0.08, frequency_hz=440.0)
    source_manifest_path, stem_paths = _write_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.08, frequency_hz=440.0)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError) as exc_info:
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        )

    error_text = str(exc_info.value)
    assert "long-tail model router rejected all production candidates" in error_text
    assert "proxy_copy_correlation_too_high" in error_text
    manifest = json.loads((output_dir / "querysep_long_tail_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rejected"
    assert manifest["failure"]["router"]["status"] == "rejected"


def test_long_tail_quality_gate_rejects_windowed_proxy_leakage(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.25, frequency_hz=220.0, duration_sec=20.0)
    _write_windowed_proxy_leakage_audio(
        output_path,
        proxy_frequency_hz=440.0,
        independent_frequency_hz=660.0,
    )
    source_manifest_path, stem_paths = _write_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.08, frequency_hz=440.0, duration_sec=20.0)

    report = querysep_long_tail_build._evaluate_long_tail_quality(
        target_stem="strings_orchestra",
        output_path=output_path,
        source_audio_path=audio_path,
        build_from_paths=querysep_long_tail_build._resolve_build_from_paths(
            explicit_build_from_paths=None,
            source_stem_manifest_path=source_manifest_path,
        ),
        thresholds=querysep_long_tail_build._build_quality_thresholds(),
    )

    assert report["status"] == "rejected"
    other_report = report["metrics"]["proxy_references"]["other"]
    assert other_report["abs_correlation"] < 0.85
    assert other_report["windowed_correlation"]["max_active_abs_correlation"] > 0.55
    assert any(
        finding["label"] == "windowed_proxy_correlation_too_high"
        for finding in report["findings"]
    )


def test_long_tail_quality_gate_rejects_possible_other_proxy_leakage(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.25, frequency_hz=220.0, duration_sec=20.0)
    _write_two_tone_audio(
        output_path,
        first_amplitude=0.04,
        first_frequency_hz=440.0,
        second_amplitude=0.069282,
        second_frequency_hz=660.0,
        duration_sec=20.0,
    )
    source_manifest_path, stem_paths = _write_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.08, frequency_hz=440.0, duration_sec=20.0)

    report = querysep_long_tail_build._evaluate_long_tail_quality(
        target_stem="strings_orchestra",
        output_path=output_path,
        source_audio_path=audio_path,
        build_from_paths=querysep_long_tail_build._resolve_build_from_paths(
            explicit_build_from_paths=None,
            source_stem_manifest_path=source_manifest_path,
        ),
        thresholds=querysep_long_tail_build._build_quality_thresholds(),
    )

    assert report["status"] == "rejected"
    other_report = report["metrics"]["proxy_references"]["other"]
    assert 0.42 < other_report["abs_correlation"] < 0.85
    assert other_report["windowed_correlation"]["max_active_abs_correlation"] < 0.55
    assert any(
        finding["label"] == "possible_other_proxy_leakage"
        for finding in report["findings"]
    )


def test_build_querysep_long_tail_stems_selects_proxy_decorrelated_candidate_for_windowed_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.25, frequency_hz=220.0, duration_sec=20.0)
    _write_windowed_proxy_leakage_audio(
        output_path,
        proxy_frequency_hz=440.0,
        independent_frequency_hz=660.0,
    )
    source_manifest_path, stem_paths = _write_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.08, frequency_hz=440.0, duration_sec=20.0)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    selected = manifest["stems"]["strings_orchestra"]
    assert selected["status"] == "accepted"
    assert selected["selected_candidate_id"] == "banquet_byoq_primary_proxy_decorrelated"
    assert selected["selected_candidate_type"] == "windowed_proxy_decorrelated_candidate"
    assert selected["selected_output_origin"] == "windowed_proxy_decorrelated_candidate"
    assert selected["quality_gate"]["status"] == "pass"
    candidates = selected["candidates"]
    primary_candidate = next(candidate for candidate in candidates if candidate["candidate_id"] == "banquet_byoq_primary")
    repaired_candidate = next(
        candidate for candidate in candidates if candidate["candidate_id"] == "banquet_byoq_primary_proxy_decorrelated"
    )
    assert primary_candidate["quality_gate"]["status"] == "rejected"
    assert any(
        finding["label"] == "windowed_proxy_correlation_too_high"
        for finding in primary_candidate["quality_gate"]["findings"]
    )
    assert repaired_candidate["metadata"]["source_candidate_id"] == "banquet_byoq_primary"
    assert repaired_candidate["metadata"]["source_candidate_type"] == "banquet_byoq"
    assert repaired_candidate["metadata"]["repair_window_count"] > 0
    assert repaired_candidate["metadata"]["repaired_windows"][0]["proxy_stem"] == "other"
    other_report = selected["quality_gate"]["metrics"]["proxy_references"]["other"]
    assert other_report["windowed_correlation"]["max_active_abs_correlation"] <= 0.55
    assert (
        other_report["windowed_correlation"]["top_windows"][0]["output_rms_dbfs"]
        >= manifest["quality_gate"]["thresholds"]["min_windowed_proxy_rms_dbfs"]
    )


def test_build_querysep_long_tail_stems_selects_harmonic_residual_query_candidate_when_primary_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    primary_output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    harmonic_query_output_path = (
        output_dir
        / "long_tail_querysep_harmonic_residual_query"
        / "01_strings_orchestra"
        / "strings_orchestra.wav"
    )
    _write_audio(audio_path, amplitude=0.25)
    _write_audio(primary_output_path, amplitude=0.001)
    _write_sine_audio(harmonic_query_output_path, amplitude=0.08, frequency_hz=660.0)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    captured_payloads: list[dict] = []

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        captured_payloads.append(payload)
        if payload["phase_id"] == "long_tail_querysep_harmonic_residual_query":
            return {
                "status": "completed",
                "named_outputs": {
                    "strings_orchestra": (
                        "/querysep/out/long_tail_querysep_harmonic_residual_query/"
                        "01_strings_orchestra/strings_orchestra.wav"
                    )
                },
                "metadata": {"routing_mode": "banquet_byoq"},
            }
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    selected = manifest["stems"]["strings_orchestra"]
    selected_candidate = next(
        candidate
        for candidate in selected["candidates"]
        if candidate["candidate_id"] == "banquet_byoq_harmonic_residual_query"
    )

    assert len(captured_payloads) == 2
    assert captured_payloads[1]["raw_step"]["candidate_type"] == "banquet_byoq_harmonic_residual_query"
    assert captured_payloads[1]["raw_step"]["query_mode"] == "harmonic_residual_single"
    assert "harmonic_residual" in captured_payloads[1]["build_from_paths"]
    assert captured_payloads[1]["build_from_paths"]["harmonic_residual"].startswith(
        "/querysep/out/router_auxiliary/strings_orchestra/harmonic_residual_query/"
    )
    assert captured_payloads[1]["output_root"] == "/querysep/out"
    assert selected["status"] == "accepted"
    assert selected["selected_candidate_id"] == "banquet_byoq_harmonic_residual_query"
    assert selected["selected_candidate_type"] == "banquet_byoq_harmonic_residual_query"
    assert selected["selected_output_origin"] == "banquet_byoq_harmonic_residual_query"
    assert selected["selected_output_path"] == str(harmonic_query_output_path)
    assert selected["quality_gate"]["status"] == "pass"
    assert selected_candidate["metadata"]["query_auxiliary_bus"]["status"] == "materialized"
    assert Path(selected_candidate["metadata"]["query_auxiliary_bus"]["local_output_path"]).exists()


def test_build_querysep_long_tail_stems_selects_sparse_harmonic_residual_candidate_when_banquet_is_near_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    primary_output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    harmonic_query_output_path = (
        output_dir
        / "long_tail_querysep_harmonic_residual_query"
        / "01_strings_orchestra"
        / "strings_orchestra.wav"
    )
    _write_windowed_proxy_leakage_audio(
        audio_path,
        proxy_frequency_hz=440.0,
        independent_frequency_hz=660.0,
        duration_sec=20.0,
    )
    _write_audio(primary_output_path, amplitude=0.001, duration_sec=20.0)
    _write_audio(harmonic_query_output_path, amplitude=0.00001, duration_sec=20.0)
    source_manifest_path, stem_paths = _write_dense_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.08, frequency_hz=440.0, duration_sec=20.0)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        if payload["phase_id"] == "long_tail_querysep_harmonic_residual_query":
            return {
                "status": "completed",
                "named_outputs": {
                    "strings_orchestra": (
                        "/querysep/out/long_tail_querysep_harmonic_residual_query/"
                        "01_strings_orchestra/strings_orchestra.wav"
                    )
                },
                "metadata": {"routing_mode": "banquet_byoq"},
            }
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    selected = manifest["stems"]["strings_orchestra"]
    candidates = selected["candidates"]
    sparse_candidate = next(candidate for candidate in candidates if candidate["candidate_id"] == "sparse_harmonic_residual")
    repaired_candidate = next(
        candidate for candidate in candidates if candidate["candidate_id"] == "sparse_harmonic_residual_proxy_decorrelated"
    )

    assert selected["status"] == "accepted"
    assert selected["selected_candidate_id"] == "sparse_harmonic_residual_proxy_decorrelated"
    assert selected["selected_candidate_type"] == "windowed_proxy_decorrelated_candidate"
    assert selected["selected_output_origin"] == "windowed_proxy_decorrelated_candidate"
    assert selected["primary_querysep_output_path"] == "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
    assert selected["querysep_output_path"] is None
    assert selected["quality_gate"]["status"] == "pass"
    assert manifest["fallback_used"] is False
    assert sparse_candidate["candidate_type"] == "sparse_harmonic_residual_candidate"
    assert sparse_candidate["quality_gate"]["status"] == "rejected"
    assert sparse_candidate["metadata"]["activation_reason"] == "harmonic_residual_banquet_presence_failure"
    assert sparse_candidate["metadata"]["source_auxiliary_bus_id"] == "harmonic_residual_query_bus"
    assert sparse_candidate["metadata"]["source_auxiliary_path"].endswith(
        "router_auxiliary/strings_orchestra/harmonic_residual_query/harmonic_residual.flac"
    )
    assert len(sparse_candidate["metadata"]["used_references"]) >= 4
    assert any(
        finding["label"] == "windowed_proxy_correlation_too_high"
        for finding in sparse_candidate["quality_gate"]["findings"]
    )
    assert repaired_candidate["metadata"]["source_candidate_id"] == "sparse_harmonic_residual"
    assert repaired_candidate["metadata"]["source_candidate_type"] == "sparse_harmonic_residual_candidate"
    assert repaired_candidate["metadata"]["repair_window_count"] > 0
    assert repaired_candidate["quality_gate"]["status"] == "pass"


def test_build_querysep_long_tail_stems_uses_querysep_output_root_for_harmonic_residual_query_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    primary_output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    harmonic_query_output_path = (
        output_dir
        / "long_tail_querysep_harmonic_residual_query"
        / "01_strings_orchestra"
        / "strings_orchestra.wav"
    )
    _write_audio(audio_path, amplitude=0.25)
    _write_audio(primary_output_path, amplitude=0.001)
    _write_sine_audio(harmonic_query_output_path, amplitude=0.08, frequency_hz=660.0)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    captured_payloads: list[dict] = []

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        captured_payloads.append(payload)
        if payload["phase_id"] == "long_tail_querysep_harmonic_residual_query":
            return {
                "status": "completed",
                "named_outputs": {
                    "strings_orchestra": (
                        "/querysep/custom-root/long_tail_querysep_harmonic_residual_query/"
                        "01_strings_orchestra/strings_orchestra.wav"
                    )
                },
                "metadata": {"routing_mode": "banquet_byoq"},
            }
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/custom-root/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        querysep_output_root="/querysep/custom-root",
        path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/custom-root"}],
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    selected = manifest["stems"]["strings_orchestra"]

    assert len(captured_payloads) == 2
    assert captured_payloads[0]["output_root"] == "/querysep/custom-root"
    assert captured_payloads[1]["output_root"] == "/querysep/custom-root"
    assert captured_payloads[1]["raw_step"]["query_mode"] == "harmonic_residual_single"
    assert captured_payloads[1]["build_from_paths"]["harmonic_residual"].startswith(
        "/querysep/custom-root/router_auxiliary/strings_orchestra/harmonic_residual_query/"
    )
    assert selected["selected_candidate_id"] == "banquet_byoq_harmonic_residual_query"
    assert selected["selected_output_path"] == str(harmonic_query_output_path)


def test_build_querysep_long_tail_stems_rejects_proxy_sample_rate_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.25, frequency_hz=220.0, sample_rate=44100)
    _write_sine_audio(output_path, amplitude=0.08, frequency_hz=440.0, sample_rate=44100)
    source_manifest_path, stem_paths = _write_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.08, frequency_hz=440.0, sample_rate=48000)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError) as exc_info:
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        )

    assert "proxy_sample_rate_mismatch" in str(exc_info.value)
    manifest = json.loads((output_dir / "querysep_long_tail_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rejected"
    assert (
        manifest["stems"]["strings_orchestra"]["quality_gate"]["metrics"]["proxy_references"]["other"]["status"]
        == "rejected_sample_rate_mismatch"
    )


def test_build_querysep_long_tail_stems_selects_external_candidate_when_primary_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    primary_output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    external_output_path = tmp_path / "external" / "qscnet_strings.flac"
    _write_sine_audio(audio_path, amplitude=0.3, frequency_hz=220.0)
    _write_sine_audio(primary_output_path, amplitude=0.001, frequency_hz=440.0)
    _write_sine_audio(external_output_path, amplitude=0.08, frequency_hz=660.0)
    source_manifest_path, _ = _write_source_manifest(tmp_path)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    result = querysep_long_tail_build.build_querysep_long_tail_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        build_id="build-123",
        source_stem_manifest_path=source_manifest_path,
        target_stems=["strings_orchestra"],
        querysep_worker_url="http://querysep-worker:8000",
        path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        long_tail_candidate_paths={"strings_orchestra": {"qscnet_strings": str(external_output_path)}},
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    selected = manifest["stems"]["strings_orchestra"]
    assert selected["status"] == "accepted"
    assert selected["selected_candidate_id"] == "external_qscnet_strings"
    assert selected["selected_output_origin"] == "external_model_candidate"
    assert selected["selected_output_path"] == str(external_output_path)
    assert selected["primary_querysep_output_path"] == "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
    assert selected["querysep_output_path"] is None
    assert selected["local_output_path"] == str(external_output_path)
    assert selected["router"]["accepted_candidate_count"] == 1
    primary_candidate = next(
        candidate for candidate in selected["candidates"] if candidate["candidate_id"] == "banquet_byoq_primary"
    )
    assert primary_candidate["quality_gate"]["status"] == "rejected"


def test_build_querysep_long_tail_stems_does_not_select_diagnostic_bus_sum_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    primary_output_path = output_dir / "long_tail_querysep" / "01_strings_orchestra" / "strings_orchestra.wav"
    _write_sine_audio(audio_path, amplitude=0.3, frequency_hz=220.0)
    _write_sine_audio(primary_output_path, amplitude=0.001, frequency_hz=440.0)
    source_manifest_path, stem_paths = _write_source_manifest(tmp_path)
    _write_sine_audio(stem_paths["other"], amplitude=0.06, frequency_hz=550.0)
    _write_sine_audio(stem_paths["piano"], amplitude=0.04, frequency_hz=880.0)

    def fake_post_json(url: str, payload: dict, *, timeout_seconds: int) -> dict:
        return {
            "status": "completed",
            "named_outputs": {
                "strings_orchestra": "/querysep/out/long_tail_querysep/01_strings_orchestra/strings_orchestra.wav"
            },
            "metadata": {"routing_mode": "banquet_byoq"},
        }

    monkeypatch.setattr(querysep_long_tail_build, "_post_json", fake_post_json)

    with pytest.raises(RuntimeError, match="rejected all production candidates"):
        querysep_long_tail_build.build_querysep_long_tail_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            build_id="build-123",
            source_stem_manifest_path=source_manifest_path,
            target_stems=["strings_orchestra"],
            querysep_worker_url="http://querysep-worker:8000",
            path_mappings=[{"local_prefix": str(output_dir), "querysep_prefix": "/querysep/out"}],
        )

    manifest = json.loads((output_dir / "querysep_long_tail_manifest.json").read_text(encoding="utf-8"))
    candidates = manifest["stems"]["strings_orchestra"]["candidates"]
    diagnostic = next(candidate for candidate in candidates if candidate["candidate_id"] == "diagnostic_harmonic_bus_sum")
    assert diagnostic["production_eligible"] is False
    assert manifest["failure"]["router"]["accepted_candidate_count"] == 0
