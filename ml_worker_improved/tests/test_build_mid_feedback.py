from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_mid_feedback import build_mid_feedback


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_audio(path: Path) -> Path:
    sample_rate = 16_000
    t = np.arange(sample_rate // 2, dtype=np.float32) / float(sample_rate)
    mono = (np.sin(2.0 * np.pi * 220.0 * t) * 0.03).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="FLAC", subtype="PCM_24")
    return path


def _mid_pack(tmp_path: Path, *, release_ready: bool = False, item_release_ready: bool = False) -> Path:
    root = tmp_path / "mid"
    _write_audio(root / "audio" / "m001" / "w001.flac")
    _write_audio(root / "audio" / "old" / "w001.flac")
    items = [
        {
            "schema": "minus_mid_audition_item.v1",
            "audio_path": "audio/m001/w001.flac",
            "source_alias": "m001",
            "source_kind": "candidate_pack_item",
            "stem_type": "minus",
            "window_id": "w001",
            "window_start_sec": 48.0,
            "window_duration_sec": 12.0,
            "window_reason": "middle_bleed",
            "source_audio_path": "/app/src/m001.flac",
            "source_pack_audio_path": "audio/mpr/min/m001.flac",
            "source_row": {"source_candidate_id": "long_candidate_id_kept_in_manifest"},
            "release_ready": False,
            "can_publish_as_result": False,
        },
        {
            "schema": "minus_mid_audition_item.v1",
            "audio_path": "audio/old/w001.flac",
            "source_alias": "old",
            "source_kind": "extra_diagnostic_source",
            "stem_type": "minus",
            "window_id": "w001",
            "window_start_sec": 48.0,
            "window_duration_sec": 12.0,
            "window_reason": "middle_bleed",
            "source_audio_path": "/app/src/old.flac",
            "source_manifest": "/app/src/old_manifest.json",
            "source_pack_audio_path": "",
            "source_row": {"extra_source_alias": "old", "source_manifest": "/app/src/old_manifest.json"},
            "release_ready": item_release_ready,
            "can_publish_as_result": False,
        },
    ]
    return _write_json(
        root / "pack.json",
        {
            "schema": "minus_mid_audition_pack.v1",
            "source_pack_json": "/app/lab_artifacts/lp_v43_001/pack.json",
            "source_pack_id": "lp_v43_001",
            "pack_status": "ready_for_diagnostic_listen",
            "delivery_allowed": False,
            "presentation_status": "diagnostic_audition_only",
            "release_ready": release_ready,
            "can_publish_as_result": False,
            "diagnostic_audio_exported": True,
            "item_count": len(items),
            "source_count": 2,
            "window_count": 1,
            "items": items,
            "failed_checks": [],
        },
    )


def test_build_mid_feedback_writes_rejected_minus_feedback(tmp_path: Path) -> None:
    manifest = build_mid_feedback(
        mid_pack_json=_mid_pack(tmp_path),
        output_dir=tmp_path / "fb",
        feedback_items=[
            {
                "audio_path": "audio/old/w001.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus", "residual_vocal_middle", "user_reported_unacceptable"],
                "note": "Middle has muffling and vocal bleed.",
            }
        ],
    )

    assert manifest["schema"] == "short_listen_feedback_manifest.v1"
    assert manifest["feedback_source_schema"] == "minus_mid_audition_pack.v1"
    assert manifest["overall_listen_verdict"] == "rejected"
    assert manifest["review_scope"] == "diagnostic_subset"
    assert manifest["review_coverage_status"] == "partial"
    assert manifest["feedback_item_count"] == 1
    assert manifest["failed_checks"] == []
    assert manifest["release_ready"] is False
    assert manifest["can_publish_as_result"] is False
    item = manifest["items"][0]
    assert item["audio_path"] == "audio/old/w001.flac"
    assert item["source_alias"] == "old"
    assert item["stem_type"] == "minus"
    assert item["source_manifest"] == "/app/src/old_manifest.json"
    assert item["source_row"]["judge_evidence"]["applied_windows"][0]["start_sec"] == 48.0
    assert item["source_row"]["judge_evidence"]["applied_windows"][0]["window_id"] == "w001"
    assert item["diagnostic_clip_audio_path"] == "audio/old/w001.flac"
    assert (tmp_path / "fb" / "feedback.json").exists()
    assert (tmp_path / "fb" / "summary.md").exists()
    assert all(len(path.name) <= 32 for path in (tmp_path / "fb").iterdir() if path.is_file())


def test_build_mid_feedback_blocks_unknown_audio_path(tmp_path: Path) -> None:
    manifest = build_mid_feedback(
        mid_pack_json=_mid_pack(tmp_path),
        output_dir=tmp_path / "fb",
        feedback_items=[
            {
                "audio_path": "audio/missing/w001.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus"],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert "feedback_audio_path_not_in_mid_pack:audio/missing/w001.flac" in manifest["failed_checks"]
    assert manifest["items"] == []


def test_build_mid_feedback_blocks_release_ready_pack(tmp_path: Path) -> None:
    manifest = build_mid_feedback(
        mid_pack_json=_mid_pack(tmp_path, release_ready=True),
        output_dir=tmp_path / "fb",
        feedback_items=[
            {
                "audio_path": "audio/old/w001.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus"],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert "mid_pack_release_ready_true" in manifest["failed_checks"]


def test_build_mid_feedback_blocks_release_ready_item(tmp_path: Path) -> None:
    manifest = build_mid_feedback(
        mid_pack_json=_mid_pack(tmp_path, item_release_ready=True),
        output_dir=tmp_path / "fb",
        feedback_items=[
            {
                "audio_path": "audio/old/w001.flac",
                "verdict": "rejected",
                "issue_tags": ["muffled_minus"],
            }
        ],
    )

    assert manifest["overall_listen_verdict"] == "invalid"
    assert "feedback_mid_item_release_ready_true:audio/old/w001.flac" in manifest["failed_checks"]
