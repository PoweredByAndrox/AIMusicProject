from __future__ import annotations

import json
from pathlib import Path

from tools.build_music_intelligence_stem_listen_pack import build_music_intelligence_stem_listen_pack


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-flac")
    return path


def test_stem_listen_pack_collects_accepted_vocal_and_minus(tmp_path: Path) -> None:
    track_id = "arch_1a_fresh_music_manifest_001__both_9323d5dcb8f7640b"
    vocal = _audio(tmp_path / "run" / "vocal.flac")
    minus = _audio(tmp_path / "run" / "minus.flac")
    mi_manifest = _write_json(
        tmp_path / "run" / "music_intelligence_input_manifest.json",
        {
            "schema": "music_intelligence_input_manifest.v1",
            "release_ready": False,
            "stems": [
                {
                    "track_id": track_id,
                    "stem_id": "arch_1a_fresh_music_manifest_001__both_9323d5dcb8f7640b__vocal__ec4be23983",
                    "stem_type": "vocal",
                    "audio_path": str(vocal),
                    "duration_sec": 12.0,
                    "quality_status": "accepted",
                    "analysis_allowed": True,
                    "release_ready": False,
                },
                {
                    "track_id": track_id,
                    "stem_id": "arch_1a_fresh_music_manifest_001__both_9323d5dcb8f7640b__minus__0de52d4b74",
                    "stem_type": "minus",
                    "audio_path": str(minus),
                    "duration_sec": 12.0,
                    "quality_status": "accepted",
                    "analysis_allowed": True,
                    "release_ready": False,
                },
            ],
        },
    )

    manifest = build_music_intelligence_stem_listen_pack(
        music_intelligence_manifests=[mi_manifest],
        output_dir=tmp_path / "listen",
        required_stem_types={"vocal", "minus"},
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["release_ready"] is False
    assert manifest["track_count"] == 1
    assert manifest["stem_counts"] == {"minus": 1, "vocal": 1}
    assert {item["stem_type"] for item in manifest["listen_audio_files"]} == {"vocal", "minus"}
    by_stem = {item["stem_type"]: Path(item["listen_audio_path"]) for item in manifest["listen_audio_files"]}
    assert by_stem["vocal"].parent.name == "t001"
    assert by_stem["minus"].parent.name == "t001"
    assert by_stem["vocal"].name == "vocal.flac"
    assert by_stem["minus"].name == "minus.flac"
    assert all(len(part) <= 16 for path in by_stem.values() for part in path.relative_to(tmp_path / "listen").parts)
    assert all(track_id not in str(path) for path in by_stem.values())
    assert {item["track_alias"] for item in manifest["listen_audio_files"]} == {"t001"}
    assert {item["source_track_id"] for item in manifest["listen_audio_files"]} == {track_id}
    assert all(item["semantic_review_required"] is True for item in manifest["listen_audio_files"])
    assert all(item["semantic_quality_verified"] is False for item in manifest["listen_audio_files"])
    assert all(item["review_status"] == "semantic_review_pending" for item in manifest["listen_audio_files"])
    assert manifest["semantic_review_required"] is True
    assert manifest["semantic_quality_verified"] is False
    assert manifest["review_status"] == "semantic_review_pending"
    assert "semantic_clean_ground_truth" in manifest["not_claimed"]
    assert Path(manifest["review_json"]).name == "review.json"
    assert Path(manifest["rows_jsonl"]).name == "rows.jsonl"
    review = json.loads(Path(manifest["review_json"]).read_text(encoding="utf-8"))
    assert review["review_status"] == "pending"
    assert review["semantic_quality_verified_item_count"] == 0
    assert review["pending_item_count"] == 2
    rows = [json.loads(line) for line in Path(manifest["rows_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert {row["semantic_quality_status"] for row in rows} == {"pending"}


def test_stem_listen_pack_blocks_unaccepted_required_stem(tmp_path: Path) -> None:
    vocal = _audio(tmp_path / "run" / "vocal.flac")
    mi_manifest = _write_json(
        tmp_path / "run" / "music_intelligence_input_manifest.json",
        {
            "schema": "music_intelligence_input_manifest.v1",
            "release_ready": False,
            "stems": [
                {
                    "track_id": "track-a",
                    "stem_id": "track-a-vocal",
                    "stem_type": "vocal",
                    "audio_path": str(vocal),
                    "quality_status": "accepted",
                    "analysis_allowed": True,
                    "release_ready": False,
                },
                {
                    "track_id": "track-a",
                    "stem_id": "track-a-minus",
                    "stem_type": "minus",
                    "audio_path": str(tmp_path / "run" / "minus.flac"),
                    "quality_status": "blocked",
                    "analysis_allowed": False,
                    "analysis_block_reason": "severe_vocal_bleed",
                    "release_ready": False,
                },
            ],
        },
    )

    manifest = build_music_intelligence_stem_listen_pack(
        music_intelligence_manifests=[mi_manifest],
        output_dir=tmp_path / "listen",
        required_stem_types={"vocal", "minus"},
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "required_stem_type_missing:minus" in manifest["failed_checks"]
    assert manifest["listen_audio_count"] == 0


def test_stem_listen_pack_collects_required_stems_from_aggregate_pack(tmp_path: Path) -> None:
    stems = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]
    items = []
    for stem in stems:
        source_audio = _audio(tmp_path / "aggregate" / "audio" / "noe6" / "t001" / f"{stem}.flac")
        items.append(
            {
                "schema": "aggregate_short_listen_pack_audio.v1",
                "source_alias": "noe6",
                "track_alias": "noe6",
                "audio_path": str(source_audio.relative_to(tmp_path / "aggregate")),
                "source_audio_path": str(source_audio),
                "stem_type": stem,
                "quality_status": "listen_preview_only",
                "semantic_quality_verified": False,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    extra_vocal = _audio(tmp_path / "aggregate" / "audio" / "vrep" / "v001.flac")
    items.append(
        {
            "schema": "aggregate_short_listen_pack_audio.v1",
            "source_alias": "vrep",
            "track_alias": "vrep",
            "audio_path": str(extra_vocal.relative_to(tmp_path / "aggregate")),
            "source_audio_path": str(extra_vocal),
            "stem_type": "vocal",
            "quality_status": "listen_preview_only",
            "semantic_quality_verified": False,
            "release_ready": False,
            "can_publish_as_result": False,
        }
    )
    aggregate_pack = _write_json(
        tmp_path / "aggregate" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "items": items,
        },
    )

    manifest = build_music_intelligence_stem_listen_pack(
        music_intelligence_manifests=[],
        aggregate_pack_manifests=[aggregate_pack],
        source_aliases={"noe6"},
        output_dir=tmp_path / "listen",
        required_stem_types=set(stems),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["semantic_review_required"] is True
    assert manifest["semantic_quality_verified"] is False
    assert manifest["review_status"] == "semantic_review_pending"
    assert manifest["source_aliases"] == ["noe6"]
    assert manifest["input_pack_jsons"] == [str(aggregate_pack)]
    assert manifest["stem_counts"] == {stem: 1 for stem in stems}
    assert manifest["listen_audio_count"] == 7

    by_stem = {item["stem_type"]: Path(item["listen_audio_path"]) for item in manifest["listen_audio_files"]}
    assert set(by_stem) == set(stems)
    assert by_stem["vocal"].relative_to(tmp_path / "listen").as_posix() == "audio/t001/vocal.flac"
    assert by_stem["minus"].relative_to(tmp_path / "listen").as_posix() == "audio/t001/minus.flac"
    assert all("noe6" not in path.as_posix() for path in by_stem.values())
    assert all(item["semantic_review_required"] is True for item in manifest["listen_audio_files"])
    assert all(item["semantic_quality_verified"] is False for item in manifest["listen_audio_files"])
    assert all(item["review_status"] == "semantic_review_pending" for item in manifest["listen_audio_files"])
    review = json.loads(Path(manifest["review_json"]).read_text(encoding="utf-8"))
    assert review["schema"] == "semantic_stem_quality_review.v1"
    assert review["review_status"] == "pending"
    assert review["reviewer_kind"] == "human_or_machine_pending"
    assert review["semantic_quality_verified_item_count"] == 0
    assert review["pending_item_count"] == 7
    assert review["reviewed_group_aliases"] == ["t001"]


def test_stem_listen_pack_preserves_aggregate_source_row_for_mode_specific_evidence(tmp_path: Path) -> None:
    mix = _audio(tmp_path / "src" / "mix.flac")
    source_pack = _write_json(
        tmp_path / "src" / "pack.json",
        {
            "schema": "stem_manifest_audio_pack_manifest.v1",
            "stem_manifest_json": str(tmp_path / "src" / "stem_manifest.json"),
            "release_ready": False,
        },
    )
    source_audio = _audio(tmp_path / "aggregate" / "audio" / "noe6" / "t001" / "vocal.flac")
    aggregate_pack = _write_json(
        tmp_path / "aggregate" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": False,
            "can_publish_as_result": False,
            "items": [
                {
                    "schema": "aggregate_short_listen_pack_audio.v1",
                    "source_alias": "noe6",
                    "track_alias": "noe6",
                    "audio_path": str(source_audio.relative_to(tmp_path / "aggregate")),
                    "source_audio_path": str(source_audio),
                    "source_manifest": str(source_pack),
                    "source_row": {
                        "stem_manifest_json": str(tmp_path / "src" / "stem_manifest.json"),
                        "audio_path": str(mix),
                        "source_stem_key": "vocals",
                    },
                    "stem_type": "vocal",
                    "quality_status": "listen_preview_only",
                    "semantic_quality_verified": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )

    manifest = build_music_intelligence_stem_listen_pack(
        music_intelligence_manifests=[],
        aggregate_pack_manifests=[aggregate_pack],
        source_aliases={"noe6"},
        output_dir=tmp_path / "listen",
        required_stem_types={"vocal"},
    )

    row = manifest["listen_audio_files"][0]
    assert row["source_row"]["stem_manifest_json"] == str(tmp_path / "src" / "stem_manifest.json")
    assert row["source_row"]["audio_path"] == str(mix)
    assert row["source_manifest"] == str(aggregate_pack)
    assert row["release_ready"] is False


def test_stem_listen_pack_blocks_release_ready_aggregate_pack(tmp_path: Path) -> None:
    source_audio = _audio(tmp_path / "aggregate" / "audio" / "noe6" / "t001" / "vocal.flac")
    aggregate_pack = _write_json(
        tmp_path / "aggregate" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_ready": True,
            "can_publish_as_result": False,
            "items": [
                {
                    "audio_path": str(source_audio.relative_to(tmp_path / "aggregate")),
                    "stem_type": "vocal",
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            ],
        },
    )

    manifest = build_music_intelligence_stem_listen_pack(
        music_intelligence_manifests=[],
        aggregate_pack_manifests=[aggregate_pack],
        output_dir=tmp_path / "listen",
        required_stem_types={"vocal"},
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert "input_pack_release_ready_true" in manifest["failed_checks"]
    assert manifest["listen_audio_count"] == 0
