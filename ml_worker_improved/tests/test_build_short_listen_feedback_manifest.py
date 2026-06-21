from __future__ import annotations

import json
from pathlib import Path

from tools.build_short_listen_feedback_manifest import build_short_listen_feedback_manifest


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _pack(tmp_path: Path) -> Path:
    items = [
        {
            "schema": "aggregate_short_listen_pack_audio.v1",
            "audio_path": "audio/voc/v002.flac",
            "source_alias": "voc",
            "stem_type": "vocal",
            "track_alias": "voc",
            "release_ready": False,
            "can_publish_as_result": False,
            "source_manifest": "/app/source/voc/pack.json",
            "source_audio_path": "/app/source/voc/v002.flac",
            "source_row": {"track_id": "Square a Saw - Oxygen (Tevlo Vocals Resynthesized)"},
        },
        {
            "schema": "aggregate_short_listen_pack_audio.v1",
            "audio_path": "audio/noe/minus.flac",
            "source_alias": "noe",
            "stem_type": "minus_preview",
            "track_alias": "noe",
            "release_ready": False,
            "can_publish_as_result": False,
            "source_manifest": "/app/source/noe/listen_check_manifest.json",
            "source_audio_path": "/app/source/noe/minus.flac",
            "source_row": {"track_id": "no_escape"},
        },
    ]
    return _write_json(
        tmp_path / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "pack_id": "lp_test",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "item_count": len(items),
            "items": items,
            "failed_checks": [],
        },
    )


def test_build_short_listen_feedback_manifest_writes_rejected_item_issues(tmp_path: Path) -> None:
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "feedback",
        feedback_items=[
            {
                "audio_path": "audio/voc/v002.flac",
                "verdict": "rejected",
                "issue_tags": ["bad_vocal_quality"],
                "note": "Rejected by listener.",
            },
            {
                "audio_path": "audio/noe/minus.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus", "residual_vocal_middle"],
                "note": "Middle section has vocal and muffling.",
            },
        ],
    )

    assert manifest["overall_listen_verdict"] == "rejected"
    assert manifest["feedback_item_count"] == 2
    assert manifest["failed_checks"] == []
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    assert manifest["items"][0]["source_alias"] == "voc"
    assert manifest["items"][1]["stem_type"] == "minus_preview"
    assert "audio/noe/minus.flac" in manifest["rejected_audio_paths"]
    assert (tmp_path / "feedback" / "feedback.json").exists()
    assert (tmp_path / "feedback" / "summary.md").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "feedback").iterdir() if path.is_file())


def test_build_short_listen_feedback_manifest_blocks_unknown_audio_path(tmp_path: Path) -> None:
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "feedback",
        feedback_items=[
            {
                "audio_path": "audio/nope/missing.flac",
                "verdict": "rejected",
                "issue_tags": ["bad_vocal_quality"],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert "feedback_audio_path_not_in_pack:audio/nope/missing.flac" in manifest["failed_checks"]
    assert manifest["items"] == []


def test_build_short_listen_feedback_manifest_blocks_empty_accepted_review(tmp_path: Path) -> None:
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "feedback",
        feedback_items=[],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert manifest["review_coverage_status"] == "none"
    assert "feedback_review_empty" in manifest["failed_checks"]


def test_build_short_listen_feedback_manifest_marks_partial_accepted_review_invalid(tmp_path: Path) -> None:
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "feedback",
        feedback_items=[
            {
                "audio_path": "audio/voc/v002.flac",
                "verdict": "accepted",
                "issue_tags": [],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert manifest["review_coverage_status"] == "partial"
    assert "feedback_accepted_review_incomplete" in manifest["failed_checks"]


def test_build_short_listen_feedback_manifest_canonicalizes_windows_paths(tmp_path: Path) -> None:
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "feedback",
        feedback_items=[
            {
                "audio_path": "audio\\noe\\minus.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus"],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "rejected"
    assert manifest["items"][0]["audio_path"] == "audio/noe/minus.flac"
    assert manifest["rejected_audio_paths"] == ["audio/noe/minus.flac"]


def test_build_short_listen_feedback_manifest_blocks_duplicate_audio_paths(tmp_path: Path) -> None:
    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=_pack(tmp_path),
        output_dir=tmp_path / "feedback",
        feedback_items=[
            {
                "audio_path": "audio/noe/minus.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus"],
            },
            {
                "audio_path": "audio\\noe\\minus.flac",
                "verdict": "rejected",
                "issue_tags": ["residual_vocal_middle"],
            },
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert "feedback_duplicate_audio_path:audio/noe/minus.flac" in manifest["failed_checks"]
    assert manifest["items"] == []


def test_build_short_listen_feedback_manifest_blocks_release_ready_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["release_ready"] = True
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = build_short_listen_feedback_manifest(
        pack_manifest_json=pack,
        output_dir=tmp_path / "feedback",
        feedback_items=[
            {
                "audio_path": "audio/noe/minus.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus"],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert "pack_release_ready_true" in manifest["failed_checks"]
