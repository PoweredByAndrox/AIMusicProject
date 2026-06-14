from __future__ import annotations

import json
from pathlib import Path

from tools.artifact_status_contract import can_promote_to_listen_result, status_view_from_manifest
from tools.build_no_reference_full_stem_pack_manifest import build_manifest, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def accepted_master_manifest(track_id: str = "track-a") -> dict:
    return {
        "manifest_type": "no_reference_vocal_minus_master_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "track_id": track_id,
        "source_mix_path": "/tmp/mix.wav",
        "selected_vocal_path": "/tmp/vocal.wav",
        "selected_minus_path": "/tmp/minus.wav",
        "pair_decision": "pair_compatible",
        "pair_compatibility_score": 0.91,
        "pair_window_count": 5,
        "accepted_stem_count": 2,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {"reference_required": False, "training_started": False, "models_changed": False},
    }


def accepted_stem_row(
    stem_type: str,
    *,
    track_id: str = "track-a",
    semantic_verified: bool = True,
    source_listen_manifest_path: str | None = None,
) -> dict:
    return {
        "track_id": track_id,
        "stem_type": stem_type,
        "model_name": "htdemucs_6s",
        "selected_audio_path": f"/tmp/{stem_type}.flac",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "selected_audio_exists": True,
        "semantic_quality_verified": semantic_verified,
        "semantic_quality_status": "proxy_semantic_pass" if semantic_verified else "not_verified",
        "stem_semantic_blockers": [],
        "stem_semantic_score": 0.88,
        "next_action": "accepted_for_listen_pack_not_release",
        "source_listen_manifest": {
            "path": source_listen_manifest_path or f"/tmp/{stem_type}_manifest.json",
            "sha256": f"sha-{stem_type}",
        },
        "safety": {"reference_required": False, "training_started": False, "models_changed": False},
    }


def stem_matrix(*rows: dict) -> dict:
    return {
        "matrix_type": "no_reference_stem_candidate_score_matrix",
        "score_mode": "stem",
        "release_ready": False,
        "ranked_candidates": list(rows),
        "summary": {
            "candidate_count": len(rows),
            "accepted_listen_preview_count": sum(1 for row in rows if row.get("delivery_allowed") is True),
            "release_ready": False,
        },
    }


def required_stem_rows() -> list[dict]:
    return [accepted_stem_row(stem) for stem in ("bass", "drums", "melody", "other")]


def source_listen_manifest(tmp_path: Path, stem_type: str, *, source_audio_path: str) -> str:
    path = tmp_path / f"{stem_type}_listen_manifest.json"
    write_json(
        path,
        {
            "manifest_type": "no_reference_stem_listen_preview",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "stem_type": stem_type,
            "track_id": "matrix-track-alias",
            "source_audio_path": source_audio_path,
            "selected_audio_path": f"/tmp/{stem_type}.flac",
            "semantic_quality_verified": True,
            "safety": {"reference_required": False, "training_started": False, "models_changed": False},
        },
    )
    return str(path)


def test_full_stem_pack_accepts_master_and_matrix_selected_stems(tmp_path: Path) -> None:
    master_path = write_json(tmp_path / "master.json", accepted_master_manifest())
    matrix_path = write_json(tmp_path / "stem_matrix.json", stem_matrix(*required_stem_rows()))

    manifest = build_manifest(master_manifest_json=master_path, stem_score_matrix_json=matrix_path)

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "full_stem_pack"
    assert manifest["accepted_stem_count"] == 6
    assert manifest["selected_stems"]["vocal"]["path"] == "/tmp/vocal.wav"
    assert manifest["selected_stems"]["minus"]["path"] == "/tmp/minus.wav"
    assert manifest["selected_stems"]["melody"]["path"] == "/tmp/melody.flac"
    assert manifest["source_master_manifest"]["path"] == str(master_path)
    assert manifest["source_stem_score_matrix"]["path"] == str(matrix_path)
    assert manifest["release_ready"] is False
    assert manifest["audio_copied"] is False
    assert manifest["audio_exported"] is False


def test_full_stem_pack_blocks_missing_required_stem(tmp_path: Path) -> None:
    rows = [accepted_stem_row(stem) for stem in ("bass", "drums", "other")]
    manifest = build_manifest(
        master_manifest_json=write_json(tmp_path / "master.json", accepted_master_manifest()),
        stem_score_matrix_json=write_json(tmp_path / "stem_matrix.json", stem_matrix(*rows)),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "missing_required_stem:melody"


def test_full_stem_pack_blocks_unverified_stem_semantic_quality(tmp_path: Path) -> None:
    rows = required_stem_rows()
    rows[2] = accepted_stem_row("melody", semantic_verified=False)
    manifest = build_manifest(
        master_manifest_json=write_json(tmp_path / "master.json", accepted_master_manifest()),
        stem_score_matrix_json=write_json(tmp_path / "stem_matrix.json", stem_matrix(*rows)),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "stem_semantic_quality_not_verified:melody"


def test_full_stem_pack_accepts_source_manifest_track_identity_when_track_ids_drift(tmp_path: Path) -> None:
    source_audio_path = "/host_testing_stuff_music/same-track.mp3"
    rows = [
        accepted_stem_row(
            stem,
            track_id="matrix-track-alias",
            source_listen_manifest_path=source_listen_manifest(tmp_path, stem, source_audio_path=source_audio_path),
        )
        for stem in ("bass", "drums", "melody", "other")
    ]

    manifest = build_manifest(
        master_manifest_json=write_json(
            tmp_path / "master.json",
            accepted_master_manifest(track_id="master-track-alias") | {"source_mix_path": source_audio_path},
        ),
        stem_score_matrix_json=write_json(tmp_path / "stem_matrix.json", stem_matrix(*rows)),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["track_identity_match_method"] == "source_listen_manifest_source_audio_path"
    assert manifest["selected_stems"]["melody"]["track_identity_match_method"] == "source_listen_manifest_source_audio_path"


def test_full_stem_pack_blocks_track_id_drift_without_matching_source_manifest(tmp_path: Path) -> None:
    rows = [
        accepted_stem_row(
            stem,
            track_id="matrix-track-alias",
            source_listen_manifest_path=source_listen_manifest(tmp_path, stem, source_audio_path="/host_testing_stuff_music/other-track.mp3"),
        )
        for stem in ("bass", "drums", "melody", "other")
    ]

    manifest = build_manifest(
        master_manifest_json=write_json(
            tmp_path / "master.json",
            accepted_master_manifest(track_id="master-track-alias") | {"source_mix_path": "/host_testing_stuff_music/same-track.mp3"},
        ),
        stem_score_matrix_json=write_json(tmp_path / "stem_matrix.json", stem_matrix(*rows)),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "track_id_mismatch:bass"


def test_full_stem_pack_cli_writes_promotable_manifest(tmp_path: Path) -> None:
    output_manifest = tmp_path / "full_stem_pack.json"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--master-manifest-json",
                str(write_json(tmp_path / "master.json", accepted_master_manifest())),
                "--stem-score-matrix-json",
                str(write_json(tmp_path / "stem_matrix.json", stem_matrix(*required_stem_rows()))),
                "--output-manifest-json",
                str(output_manifest),
                "--summary-md",
                str(summary_md),
            ]
        )
        == 0
    )
    assert can_promote_to_listen_result(output_manifest) is True
    assert status_view_from_manifest(output_manifest).stem_type == "full_stem_pack"
    assert "delivery_status: accepted" in summary_md.read_text(encoding="utf-8")
