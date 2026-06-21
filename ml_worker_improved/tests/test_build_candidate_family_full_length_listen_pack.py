from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_candidate_family_full_length_listen_pack import build_candidate_family_full_length_listen_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _audio(path: Path, seconds: float, sample_rate: int = 8_000, scale: float = 0.1) -> Path:
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    mono = scale * np.sin(2.0 * np.pi * 220.0 * t)
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def test_full_length_listen_pack_uses_source_duration_not_short_target(tmp_path: Path) -> None:
    track_id = "Square a Saw - Oxygen / Tevlo Vocals Resynthesized Long Name"
    scnet = _audio(tmp_path / "track-a" / "scnet.flac", 4.0, scale=0.1)
    melband = _audio(tmp_path / "track-a" / "melband.flac", 4.0, scale=0.2)
    target = _audio(tmp_path / "track-a" / "target_first_second.flac", 1.0, scale=0.15)
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "track-a-scnet",
                "track_id": track_id,
                "stem_type": "vocal",
                "source_candidate_family": "scnet",
                "source_audio_path": str(scnet),
                "target_audio_path": str(target),
                "target_audio_sha256": "short-target",
                "trainable": True,
                "reference_verified": True,
                "release_ready": False,
            },
            {
                "id": "track-a-melband",
                "track_id": track_id,
                "stem_type": "vocal",
                "source_candidate_family": "melband",
                "source_audio_path": str(melband),
                "target_audio_path": str(target),
                "target_audio_sha256": "short-target",
                "trainable": True,
                "reference_verified": True,
                "release_ready": False,
            },
        ],
    )
    adapter = _write_json(
        tmp_path / "adapter.json",
        {
            "schema": "candidate_family_blend_adapter.v1",
            "config_id": "v32_scnet_melband_blend_step005_with_a_needlessly_long_identifier",
            "stem": "vocal",
            "families": ["scnet", "melband"],
            "weights": {"scnet": 0.5, "melband": 0.5},
            "release_ready": False,
            "delivery_allowed": False,
        },
    )

    manifest = build_candidate_family_full_length_listen_pack(
        audio_training_examples_jsonl=examples,
        adapter_json=adapter,
        output_dir=tmp_path / "listen",
        stem="vocal",
        track_ids={track_id},
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["release_ready"] is False
    assert manifest["listen_audio_count"] == 1
    listen_path = Path(manifest["listen_audio_files"][0]["listen_audio_path"])
    assert listen_path.parent.name == "audio"
    assert listen_path.name == "v001.flac"
    assert len(listen_path.name) <= 16
    assert "Square" not in listen_path.name
    assert "v32_scnet" not in listen_path.name
    assert manifest["listen_audio_files"][0]["artifact_alias"] == "v001"
    assert manifest["listen_audio_files"][0]["source_track_id"] == track_id
    assert round(_duration(listen_path), 3) == 4.0
    assert manifest["listen_audio_files"][0]["target_duration_sec"] == 1.0
    assert manifest["listen_audio_files"][0]["source_duration_sec"] == 4.0
    assert manifest["listen_audio_files"][0]["duration_sec"] == 4.0


def test_full_length_listen_pack_blocks_release_ready_source_row(tmp_path: Path) -> None:
    source = _audio(tmp_path / "track-a" / "scnet.flac", 4.0)
    examples = _write_jsonl(
        tmp_path / "examples.jsonl",
        [
            {
                "id": "track-a-scnet",
                "track_id": "track-a",
                "stem_type": "vocal",
                "source_candidate_family": "scnet",
                "source_audio_path": str(source),
                "target_audio_path": str(source),
                "trainable": True,
                "reference_verified": True,
                "release_ready": True,
            }
        ],
    )
    adapter = _write_json(
        tmp_path / "adapter.json",
        {"schema": "candidate_family_blend_adapter.v1", "stem": "vocal", "families": ["scnet"], "weights": {"scnet": 1.0}},
    )

    manifest = build_candidate_family_full_length_listen_pack(
        audio_training_examples_jsonl=examples,
        adapter_json=adapter,
        output_dir=tmp_path / "listen",
        stem="vocal",
        track_ids={"track-a"},
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "release_ready_input_row" in manifest["failed_checks"]
