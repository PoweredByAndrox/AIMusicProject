from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.build_aggregate_short_listen_pack import build_aggregate_short_listen_pack


REQUIRED_FULL_PACK_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def _write_audio(path: Path, *, duration_sec: float = 0.25) -> Path:
    sample_rate = 16_000
    frames = int(sample_rate * duration_sec)
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    mono = (np.sin(2.0 * np.pi * 220.0 * t) * 0.05).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _source_pack(tmp_path: Path, *, release_ready: bool = False) -> Path:
    vocal = _write_audio(tmp_path / "source" / "voc" / "v001.flac")
    stem = _write_audio(tmp_path / "source" / "six" / "t001" / "vocal.flac")
    return _write_json(
        tmp_path / "source_pack.json",
        {
            "schema": "short_source_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "listen_audio_files": [
                {
                    "artifact_relative_path": "audio/v001.flac",
                    "listen_audio_path": str(vocal),
                    "stem_type": "vocal",
                    "track_alias": "t001",
                    "quality_status": "listen_preview_only",
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "artifact_relative_path": "audio/t001/vocal.flac",
                    "listen_audio_path": str(stem),
                    "stem_type": "vocal",
                    "track_alias": "t001",
                    "quality_status": "listen_preview_only_not_semantic_verified",
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )


def _minus_listen_manifest(tmp_path: Path) -> Path:
    minus = _write_audio(tmp_path / "minus_src" / "minus.flac")
    return _write_json(
        tmp_path / "minus" / "listen_check_manifest.json",
        {
            "schema": "listen_check_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "listenable": True,
            "deliverable": False,
            "artifact_kind": "minus_preview",
            "audio_path": str(minus),
            "duration_sec": 0.25,
            "candidate": {
                "full_result_decision": "accept_candidate",
                "full_result_vocal_leakage_in_minus_score": 0.07,
                "max_vocal_leakage_in_minus_score": 0.14,
            },
            "safety": {
                "release_ready": False,
                "can_publish_as_result": False,
                "training_started": False,
                "thresholds_relaxed": False,
            },
        },
    )


def _no_reference_master_manifest(tmp_path: Path) -> Path:
    vocal = _write_audio(tmp_path / "source" / "vocal_candidate_with_raw_route_name.flac")
    minus = _write_audio(tmp_path / "source" / "minus_judge_guided_leakage_repair.flac")
    return _write_json(
        tmp_path / "master" / "manifest.json",
        {
            "manifest_type": "no_reference_vocal_minus_master_listen_preview",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "stem_type": "both",
            "accepted_stem_count": 2,
            "accepted_listen_preview_rows": [
                {
                    "track_id": "track-a",
                    "stem_type": "vocal",
                    "candidate_id": "vocal-candidate",
                    "selected_audio_path": str(vocal),
                    "source_manifest": "vocal_manifest.json",
                },
                {
                    "track_id": "track-a",
                    "stem_type": "minus",
                    "candidate_id": "minus-candidate",
                    "selected_audio_path": str(minus),
                    "source_manifest": "minus_manifest.json",
                },
            ],
            "safety": {
                "release_ready": False,
                "can_publish_as_result": False,
                "training_started": False,
                "thresholds_relaxed": False,
            },
        },
    )


def test_aggregate_short_listen_pack_copies_sources_with_short_aliases(tmp_path: Path) -> None:
    manifest = build_aggregate_short_listen_pack(
        sources=[
            ("voc", _source_pack(tmp_path)),
            ("noe", _minus_listen_manifest(tmp_path)),
        ],
        output_dir=tmp_path / "out",
        pack_id="lp_test",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["item_count"] == 3
    paths = [item["audio_path"] for item in manifest["items"]]
    assert paths == [
        "audio/voc/v001.flac",
        "audio/voc/t001/vocal.flac",
        "audio/noe/minus.flac",
    ]
    for relative in paths:
        assert (tmp_path / "out" / relative).exists()
        assert all(len(segment) <= 32 for segment in Path(relative).parts)
    assert (tmp_path / "out" / "pack.json").exists()
    assert (tmp_path / "out" / "coverage.json").exists()
    assert (tmp_path / "out" / "selected.jsonl").exists()
    assert (tmp_path / "out" / "progress.jsonl").exists()
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text(encoding="utf-8"))
    assert {group["group_alias"] for group in coverage["stem_groups"]} == {"voc", "noe"}
    assert coverage["release_ready"] is False


def test_aggregate_short_listen_pack_copies_no_reference_master_rows_with_short_names(tmp_path: Path) -> None:
    manifest = build_aggregate_short_listen_pack(
        sources=[("mst", _no_reference_master_manifest(tmp_path))],
        output_dir=tmp_path / "out",
        pack_id="lp_master",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["item_count"] == 2
    paths = [item["audio_path"] for item in manifest["items"]]
    assert paths == [
        "audio/mst/vocal.flac",
        "audio/mst/minus.flac",
    ]
    for relative in paths:
        assert (tmp_path / "out" / relative).exists()
        assert all(len(segment) <= 32 for segment in Path(relative).parts)
    assert "vocal_candidate_with_raw_route_name" not in "/".join(paths)
    assert "minus_judge_guided_leakage_repair" not in "/".join(paths)
    assert {item["stem_type"] for item in manifest["items"]} == {"vocal", "minus"}
    assert all(item["release_ready"] is False for item in manifest["items"])


def test_aggregate_short_listen_pack_full_pack_contract_blocks_missing_stems(tmp_path: Path) -> None:
    manifest = build_aggregate_short_listen_pack(
        sources=[("mst", _no_reference_master_manifest(tmp_path))],
        output_dir=tmp_path / "out",
        pack_id="lp_full",
        required_stems=REQUIRED_FULL_PACK_STEMS,
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["item_count"] == 0
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal", "minus"]
    assert manifest["missing_required_stems"] == ["drums", "bass", "other", "guitar", "piano"]
    assert "required_stem_missing:drums" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "audio").exists()


def test_aggregate_short_listen_pack_inherits_required_stems_from_source_manifest(tmp_path: Path) -> None:
    source = _no_reference_master_manifest(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["required_stems"] = REQUIRED_FULL_PACK_STEMS
    payload["present_stems"] = ["vocal", "minus"]
    payload["missing_required_stems"] = ["drums", "bass", "other", "guitar", "piano"]
    source.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = build_aggregate_short_listen_pack(
        sources=[("mst", source)],
        output_dir=tmp_path / "out",
        pack_id="lp_full",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["required_stems"] == REQUIRED_FULL_PACK_STEMS
    assert manifest["present_stems"] == ["vocal", "minus"]
    assert manifest["missing_required_stems"] == ["drums", "bass", "other", "guitar", "piano"]
    assert "required_stem_missing:drums" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "audio").exists()


def test_aggregate_short_listen_pack_blocks_required_unverified_stem(tmp_path: Path) -> None:
    audio = _write_audio(tmp_path / "source" / "d001.flac")
    source_manifest = _write_json(
        tmp_path / "source" / "pack.json",
        {
            "schema": "stem_listen_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "required_stems": ["drums"],
            "accepted_listen_preview_rows": [
                {
                    "listen_audio_path": str(audio),
                    "artifact_relative_path": "d001.flac",
                    "stem_type": "drums",
                    "track_alias": "t001",
                    "quality_status": "listen_preview_only",
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )

    manifest = build_aggregate_short_listen_pack(
        sources=[("drm", source_manifest)],
        output_dir=tmp_path / "out",
        pack_id="lp001",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["items"] == []
    assert "required_stem_quality_not_verified:drums" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "audio").exists()


def test_aggregate_short_listen_pack_blocks_mixed_required_unverified_stem_rows(tmp_path: Path) -> None:
    verified_audio = _write_audio(tmp_path / "source" / "d001.flac")
    unverified_audio = _write_audio(tmp_path / "source" / "d002.flac")
    source_manifest = _write_json(
        tmp_path / "source" / "pack.json",
        {
            "schema": "stem_listen_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "required_stems": ["drums"],
            "accepted_listen_preview_rows": [
                {
                    "listen_audio_path": str(verified_audio),
                    "artifact_relative_path": "d001.flac",
                    "stem_type": "drums",
                    "track_alias": "t001",
                    "quality_status": "listen_preview_only",
                    "semantic_quality_verified": True,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
                {
                    "listen_audio_path": str(unverified_audio),
                    "artifact_relative_path": "d002.flac",
                    "stem_type": "drums",
                    "track_alias": "t001",
                    "quality_status": "listen_preview_only",
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                },
            ],
        },
    )

    manifest = build_aggregate_short_listen_pack(
        sources=[("drm", source_manifest)],
        output_dir=tmp_path / "out",
        pack_id="lp001",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["items"] == []
    assert "required_stem_quality_not_verified:drums" in manifest["failed_checks"]


def test_aggregate_short_listen_pack_preserves_unrequired_instrument_preview(tmp_path: Path) -> None:
    audio = _write_audio(tmp_path / "source" / "d001.flac")
    source_manifest = _write_json(
        tmp_path / "source" / "pack.json",
        {
            "schema": "stem_listen_pack.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "accepted_listen_preview_rows": [
                {
                    "listen_audio_path": str(audio),
                    "artifact_relative_path": "d001.flac",
                    "stem_type": "drums",
                    "track_alias": "t001",
                    "quality_status": "listen_preview_only",
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )

    manifest = build_aggregate_short_listen_pack(
        sources=[("drm", source_manifest)],
        output_dir=tmp_path / "out",
        pack_id="lp001",
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["item_count"] == 1
    assert manifest["required_stems"] == []


def test_aggregate_short_listen_pack_blocks_release_ready_source(tmp_path: Path) -> None:
    manifest = build_aggregate_short_listen_pack(
        sources=[("voc", _source_pack(tmp_path, release_ready=True))],
        output_dir=tmp_path / "out",
        pack_id="lp_test",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["item_count"] == 0
    assert "source_release_ready_true:voc" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "audio").exists()


def test_aggregate_short_listen_pack_blocks_source_without_delivery_contract(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "bad_listen_check.json",
        {
            "schema": "listen_check_manifest.v1",
            "artifact_kind": "minus_preview",
            "audio_path": str(_write_audio(tmp_path / "minus.flac")),
            "safety": {"release_ready": False, "can_publish_as_result": False},
        },
    )

    manifest = build_aggregate_short_listen_pack(
        sources=[("noe", source)],
        output_dir=tmp_path / "out",
        pack_id="lp_test",
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["item_count"] == 0
    assert "source_delivery_status_missing:noe" in manifest["failed_checks"]
    assert "source_delivery_allowed_missing:noe" in manifest["failed_checks"]
    assert "source_presentation_status_missing:noe" in manifest["failed_checks"]
    assert not (tmp_path / "out" / "audio").exists()


def test_aggregate_short_listen_pack_rejects_long_source_alias(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact path segment too long"):
        build_aggregate_short_listen_pack(
            sources=[("this_alias_name_is_far_too_long_for_windows", _source_pack(tmp_path))],
            output_dir=tmp_path / "out",
            pack_id="lp_test",
        )
