from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import run_mac_unified_track_pipeline as unified
from tools.run_mac_unified_track_pipeline import run_unified_pipeline


SAMPLE_RATE = 16_000


def _write(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_24")
    return path


def _stereo_tone(seconds: float, freq: float, gain: float = 0.05) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / float(SAMPLE_RATE)
    mono = gain * np.sin(2.0 * math.pi * freq * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _melband_run(root: Path) -> Path:
    vocals = _stereo_tone(1.5, 220.0)
    instrumental = _stereo_tone(1.5, 330.0)
    raw = vocals + instrumental
    v = _write(root / "vocals.flac", vocals)
    i = _write(root / "instrumental.flac", instrumental)
    raw_path = _write(root / "raw_ref.flac", raw)
    (root / "stem_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "melband_roformer_vocals",
                "source_audio_path": "/mac/source/a.flac",
                "stems": {"vocals": {"path": str(v)}, "instrumental": {"path": str(i)}},
                "artifacts": {"raw_ref_flac": str(raw_path)},
            }
        ),
        encoding="utf-8",
    )
    return root


def _stem_run(root: Path, stems: list[str]) -> Path:
    stem_map = {}
    for index, stem in enumerate(stems):
        path = _write(root / f"{stem}.flac", _stereo_tone(1.5, 110.0 + 30.0 * index))
        stem_map[stem] = {"path": str(path), "duration_sec": 1.5, "rms": 0.03, "peak": 0.05}
    (root / "stem_manifest.json").write_text(
        json.dumps({"model_id": "htdemucs_6s", "source_audio_path": "/mac/source/a.flac", "stems": stem_map}),
        encoding="utf-8",
    )
    return root


def test_unified_pipeline_runs_melband_and_stem_lanes_and_blocks_missing_stems(tmp_path: Path) -> None:
    mel_run = _melband_run(tmp_path / "mel" / "run")
    stem_run = _stem_run(tmp_path / "stem" / "run", ["vocals", "drums", "bass", "other"])
    submitted_routes: list[str] = []

    def fake_submit(*, api_url: str, source_audio_path: str, output_subdir: str, route_id: str) -> dict[str, object]:
        submitted_routes.append(route_id)
        return {"job_id": f"job{len(submitted_routes)}", "route_id": route_id, "status": "queued"}

    def fake_poll(*, api_url: str, job_id: str, poll_sec: float, timeout_sec: float) -> dict[str, object]:
        route = submitted_routes[int(job_id.replace("job", "")) - 1]
        return {
            "job_id": job_id,
            "route_id": route,
            "status": "ok",
            "run_dir": str(mel_run if route == "melband_roformer_vocals" else stem_run),
            "delivery_allowed": False,
            "presentation_status": "diagnostic_mps_api_only",
        }

    manifest = run_unified_pipeline(
        mac_api_url="http://mac.local:8765",
        source_audio_path="/mac/source/a.flac",
        output_dir=tmp_path / "out",
        mel_api_output_subdir="mac_api/u001/mel",
        stem_api_output_subdir="mac_api/u001/stm",
        submit_job=fake_submit,
        poll_job=fake_poll,
        guard_config={"score_threshold": 0.99, "window_sec": 0.25, "hop_sec": 0.125},
    )

    assert submitted_routes == ["melband_roformer_vocals", "htdemucs_6s"]
    assert manifest["profile"] == "mac_unified_track_pipeline_v1"
    assert manifest["lane_contract"]["independent_lane_winners"] is True
    assert manifest["lane_contract"]["do_not_regenerate_passed_vocal_minus_for_stem_failures"] is True
    assert manifest["delivery_allowed"] is False
    assert manifest["listen_pack_created"] is False
    assert manifest["stem_lane"]["required_stems_present"] is False
    assert "blocked_missing_required_stems" in manifest["blocker_reasons"]
    assert "guitar" in manifest["stem_lane"]["missing_required_stems"]
    assert "piano" in manifest["stem_lane"]["missing_required_stems"]
    assert (tmp_path / "out" / "progress.jsonl").exists()


def test_unified_pipeline_blocks_damaged_melband_repair(monkeypatch, tmp_path: Path) -> None:
    mel_run = tmp_path / "mel" / "run"
    mel_run.mkdir(parents=True)
    stem_run = _stem_run(tmp_path / "stem" / "run", ["vocals", "drums", "bass", "guitar", "piano", "other"])
    submitted_routes: list[str] = []

    def fake_submit(*, api_url: str, source_audio_path: str, output_subdir: str, route_id: str) -> dict[str, object]:
        submitted_routes.append(route_id)
        return {"job_id": f"job{len(submitted_routes)}", "route_id": route_id, "status": "queued"}

    def fake_poll(*, api_url: str, job_id: str, poll_sec: float, timeout_sec: float) -> dict[str, object]:
        route = submitted_routes[int(job_id.replace("job", "")) - 1]
        return {
            "job_id": job_id,
            "route_id": route,
            "status": "ok",
            "run_dir": str(mel_run if route == "melband_roformer_vocals" else stem_run),
            "delivery_allowed": False,
        }

    def fake_guard(*, melband_run_dir: Path, output_dir: Path, config=None) -> dict[str, object]:
        return {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "audio": {
                "instrumental": str(output_dir / "audio" / "i.flac"),
                "vocals": str(output_dir / "audio" / "v.flac"),
            },
            "score": {"verdict": "fail", "regions": [{"start": 0.25, "end": 0.75}]},
            "repair": {"repaired_instrumental_path": str(output_dir / "repair" / "i.flac")},
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
        }

    def fake_damage(**kwargs) -> dict[str, object]:
        return {
            "verdict": "fail",
            "blocker_reasons": ["instrumental_region_overprocessed"],
            "failed_region_count": 1,
            "delivery_allowed": False,
        }

    monkeypatch.setattr(unified, "run_pitch_guard_on_existing_melband_run", fake_guard)
    monkeypatch.setattr(unified, "run_damage_audit", fake_damage)

    manifest = run_unified_pipeline(
        mac_api_url="http://mac.local:8765",
        source_audio_path="/mac/source/a.flac",
        output_dir=tmp_path / "out",
        mel_api_output_subdir="mac_api/u002/mel",
        stem_api_output_subdir="mac_api/u002/stm",
        submit_job=fake_submit,
        poll_job=fake_poll,
    )

    assert manifest["stem_lane"]["required_stems_present"] is True
    assert manifest["melband_lane"]["damage_audit"]["verdict"] == "fail"
    assert "pitch_shifted_vocal_in_instrumental" in manifest["blocker_reasons"]
    assert "instrumental_region_overprocessed" in manifest["blocker_reasons"]
    assert manifest["delivery_allowed"] is False


def test_unified_pipeline_blocks_malformed_guard_without_damage_audit(monkeypatch, tmp_path: Path) -> None:
    mel_run = tmp_path / "mel" / "run"
    mel_run.mkdir(parents=True)
    stem_run = _stem_run(tmp_path / "stem" / "run", ["vocals", "drums", "bass", "guitar", "piano", "other"])
    submitted_routes: list[str] = []

    def fake_submit(*, api_url: str, source_audio_path: str, output_subdir: str, route_id: str) -> dict[str, object]:
        submitted_routes.append(route_id)
        return {"job_id": f"job{len(submitted_routes)}", "route_id": route_id, "status": "queued"}

    def fake_poll(*, api_url: str, job_id: str, poll_sec: float, timeout_sec: float) -> dict[str, object]:
        route = submitted_routes[int(job_id.replace("job", "")) - 1]
        return {
            "job_id": job_id,
            "route_id": route,
            "status": "ok",
            "run_dir": str(mel_run if route == "melband_roformer_vocals" else stem_run),
            "delivery_allowed": False,
        }

    def fake_guard(*, melband_run_dir: Path, output_dir: Path, config=None) -> dict[str, object]:
        return {
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "audio": {"instrumental": str(output_dir / "audio" / "i.flac")},
            "score": {"verdict": "fail", "regions": [{"start": 0.25, "end": 0.75}]},
            "blocker_reasons": ["pitch_shifted_vocal_in_instrumental"],
        }

    monkeypatch.setattr(unified, "run_pitch_guard_on_existing_melband_run", fake_guard)

    manifest = run_unified_pipeline(
        mac_api_url="http://mac.local:8765",
        source_audio_path="/mac/source/a.flac",
        output_dir=tmp_path / "out",
        mel_api_output_subdir="mac_api/u003/mel",
        stem_api_output_subdir="mac_api/u003/stm",
        submit_job=fake_submit,
        poll_job=fake_poll,
    )

    assert manifest["stem_lane"]["required_stems_present"] is True
    assert manifest["melband_lane"]["damage_audit"]["verdict"] == "blocked_unknown"
    assert "damage_audit_missing_repair_manifest" in manifest["blocker_reasons"]
    assert manifest["delivery_allowed"] is False


def test_unified_pipeline_missing_run_dir_writes_blocked_manifest(tmp_path: Path) -> None:
    submitted_routes: list[str] = []

    def fake_submit(*, api_url: str, source_audio_path: str, output_subdir: str, route_id: str) -> dict[str, object]:
        submitted_routes.append(route_id)
        return {"job_id": f"job{len(submitted_routes)}", "route_id": route_id, "status": "queued"}

    def fake_poll(*, api_url: str, job_id: str, poll_sec: float, timeout_sec: float) -> dict[str, object]:
        route = submitted_routes[int(job_id.replace("job", "")) - 1]
        return {
            "job_id": job_id,
            "route_id": route,
            "status": "ok",
            "run_dir": str(tmp_path / route / "missing"),
            "delivery_allowed": False,
        }

    manifest = run_unified_pipeline(
        mac_api_url="http://mac.local:8765",
        source_audio_path="/mac/source/a.flac",
        output_dir=tmp_path / "out",
        mel_api_output_subdir="mac_api/u004/mel",
        stem_api_output_subdir="mac_api/u004/stm",
        submit_job=fake_submit,
        poll_job=fake_poll,
    )

    assert manifest["delivery_status"] == "blocked"
    assert "mac_melband_run_dir_missing" in manifest["blocker_reasons"]
    assert "mac_stem_run_dir_missing" in manifest["blocker_reasons"]
    assert manifest["melband_lane"]["guard_manifest"]["delivery_status"] == "blocked"
    assert manifest["stem_lane"]["required_stems_present"] is False
    assert (tmp_path / "out" / "manifest.json").exists()
