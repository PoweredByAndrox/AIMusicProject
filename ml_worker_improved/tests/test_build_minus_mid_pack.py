from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_minus_mid_pack import build_minus_mid_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_audio(path: Path, *, duration_sec: float = 4.0) -> Path:
    sample_rate = 16_000
    samples = int(sample_rate * duration_sec)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    mono = (np.sin(2.0 * np.pi * 220.0 * t) * 0.04).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _pack(tmp_path: Path, *, release_ready: bool = False, item_release_ready: bool = False) -> Path:
    root = tmp_path / "lp"
    m001 = _write_audio(root / "audio" / "mpr" / "min" / "m001.flac")
    m002 = _write_audio(root / "audio" / "mpr" / "min" / "m002.flac")
    _write_audio(root / "audio" / "vrep" / "sub" / "v001.flac")
    return _write_json(
        root / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "failed_checks": [],
            "items": [
                {
                    "audio_path": "audio/mpr/min/m001.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "track_alias": "mpr",
                    "release_ready": item_release_ready,
                    "can_publish_as_result": False,
                    "source_audio_path": str(m001),
                    "source_row": {"source_candidate_id": "very_long_candidate_id_kept_in_manifest_only"},
                },
                {
                    "audio_path": "audio/mpr/min/m002.flac",
                    "source_alias": "mpr",
                    "stem_type": "minus",
                    "track_alias": "mpr",
                    "release_ready": False,
                    "can_publish_as_result": False,
                    "source_audio_path": str(m002),
                    "source_row": {"source_candidate_id": "second_long_candidate_id_kept_in_manifest_only"},
                },
                {
                    "audio_path": "audio/vrep/sub/v001.flac",
                    "source_alias": "vrep",
                    "stem_type": "vocal",
                    "track_alias": "vrep",
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )


def test_build_minus_mid_pack_writes_short_diagnostic_clips(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    old = _write_audio(tmp_path / "old_source" / "long_rejected_minus_name.flac")
    old_manifest = _write_json(
        tmp_path / "old_source" / "pack.json",
        {"schema": "short_minus_phase_repair_selection_pack_manifest.v1", "release_ready": False},
    )

    manifest = build_minus_mid_pack(
        candidate_pack_json=pack,
        output_dir=tmp_path / "mid",
        windows=[
            {"window_id": "w001", "start_sec": 0.25, "duration_sec": 0.5, "reason": "middle_bleed"},
            {"window_id": "w002", "start_sec": 1.0, "duration_sec": 0.5, "reason": "late_detail"},
        ],
        source_alias="mpr",
        stem_type="minus",
        extra_sources=[("old", old)],
        extra_source_manifests=[("old", old_manifest)],
        handoff_playlist_m3u=tmp_path / "handoff" / "mid.m3u",
        command_args=["--example"],
    )

    assert manifest["schema"] == "minus_mid_audition_pack.v1"
    assert manifest["pack_status"] == "ready_for_diagnostic_listen"
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["diagnostic_audio_exported"] is True
    assert manifest["item_count"] == 6
    assert manifest["source_count"] == 3
    assert manifest["window_count"] == 2
    assert manifest["failed_checks"] == []
    assert manifest["source_pack_json"] == str(pack)

    items_path = tmp_path / "mid" / "items.jsonl"
    rows = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
    assert [row["audio_path"] for row in rows] == [
        "audio/m001/w001.flac",
        "audio/m001/w002.flac",
        "audio/m002/w001.flac",
        "audio/m002/w002.flac",
        "audio/old/w001.flac",
        "audio/old/w002.flac",
    ]
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(row["release_ready"] is False for row in rows)
    assert all(row["source_audio_sha256"] for row in rows)
    assert rows[0]["source_row"]["source_candidate_id"] == "very_long_candidate_id_kept_in_manifest_only"
    assert rows[-1]["source_manifest"] == str(old_manifest)
    assert rows[-1]["source_row"]["source_manifest"] == str(old_manifest)

    for row in rows:
        path = Path(row["audio_path"])
        assert all(len(segment) <= 32 for segment in path.parts)
        assert (tmp_path / "mid" / path).exists()

    playlist = (tmp_path / "mid" / "playlist.m3u").read_text(encoding="utf-8")
    assert "audio/m001/w001.flac" in playlist
    assert "very_long_candidate_id" not in playlist
    assert (tmp_path / "handoff" / "mid.m3u").exists()
    assert "../mid/audio/m001/w001.flac" in (tmp_path / "handoff" / "mid.m3u").read_text(encoding="utf-8")


def test_build_minus_mid_pack_blocks_release_ready_pack(tmp_path: Path) -> None:
    manifest = build_minus_mid_pack(
        candidate_pack_json=_pack(tmp_path, release_ready=True),
        output_dir=tmp_path / "mid",
        windows=[{"window_id": "w001", "start_sec": 0.0, "duration_sec": 0.5, "reason": "probe"}],
        source_alias="mpr",
        stem_type="minus",
        extra_sources=[],
    )

    assert manifest["pack_status"] == "blocked"
    assert manifest["item_count"] == 0
    assert "source_pack_release_ready_true" in manifest["failed_checks"]
    assert not (tmp_path / "mid" / "audio").exists()


def test_build_minus_mid_pack_blocks_release_ready_item(tmp_path: Path) -> None:
    manifest = build_minus_mid_pack(
        candidate_pack_json=_pack(tmp_path, item_release_ready=True),
        output_dir=tmp_path / "mid",
        windows=[{"window_id": "w001", "start_sec": 0.0, "duration_sec": 0.5, "reason": "probe"}],
        source_alias="mpr",
        stem_type="minus",
        extra_sources=[],
    )

    assert manifest["pack_status"] == "blocked"
    assert "source_item_release_ready_true:m001" in manifest["failed_checks"]
