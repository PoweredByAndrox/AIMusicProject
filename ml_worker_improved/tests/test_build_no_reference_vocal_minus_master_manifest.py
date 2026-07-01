from __future__ import annotations

import json
from pathlib import Path

from tools.artifact_status_contract import can_promote_to_listen_result
from tools.build_no_reference_vocal_minus_master_manifest import build_manifest, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def accepted_vocal_manifest(track_id: str = "track-a") -> dict:
    return {
        "manifest_type": "no_reference_vocal_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "vocal",
        "track_id": track_id,
        "candidate_id": "vocal-a",
        "mix_path": "/tmp/mix.wav",
        "vocal_path": "/tmp/vocal.wav",
        "selected_vocal_path": "/tmp/vocal.wav",
        "selected_audio_path": "/tmp/vocal.wav",
        "accepted_count": 1,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {"reference_required": False, "training_started": False, "models_changed": False},
    }


def accepted_minus_manifest(track_id: str = "track-a") -> dict:
    return {
        "manifest_type": "no_reference_minus_listen_preview",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "minus",
        "track_id": track_id,
        "candidate_id": "minus-a",
        "mix_path": "/tmp/mix.wav",
        "minus_path": "/tmp/minus.wav",
        "instrumental_path": "/tmp/minus.wav",
        "selected_minus_path": "/tmp/minus.wav",
        "selected_audio_path": "/tmp/minus.wav",
        "accepted_count": 1,
        "can_mark_release_ready": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {"reference_required": False, "training_started": False, "models_changed": False},
    }


def accepted_pair_compatibility_row(track_id: str = "track-a") -> dict:
    return {
        "track_id": track_id,
        "candidate_id": "master-pair-a",
        "mix_path": "/tmp/mix.wav",
        "vocal_path": "/tmp/vocal.wav",
        "instrumental_path": "/tmp/minus.wav",
        "minus_path": "/tmp/minus.wav",
        "score_mode": "both",
        "decision": "accept_candidate",
        "pair_decision": "pair_compatible",
        "pair_compatibility_score": 0.91,
        "pair_blockers": [],
        "window_count": 5,
    }


def test_master_manifest_accepts_only_independently_accepted_vocal_and_minus(tmp_path: Path) -> None:
    vocal_path = write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest())
    minus_path = write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest())
    pair_path = write_jsonl(tmp_path / "pair_compatibility.jsonl", [accepted_pair_compatibility_row()])

    manifest = build_manifest(
        vocal_manifest_json=vocal_path,
        minus_manifest_json=minus_path,
        pair_compatibility_jsonl=pair_path,
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["stem_type"] == "both"
    assert manifest["track_id"] == "track-a"
    assert manifest["selected_vocal_path"] == "/tmp/vocal.wav"
    assert manifest["selected_minus_path"] == "/tmp/minus.wav"
    assert manifest["accepted_stem_count"] == 2
    assert manifest["pair_decision"] == "pair_compatible"
    assert manifest["pair_compatibility_score"] == 0.91
    assert manifest["source_pair_compatibility_jsonl"]["path"] == str(pair_path)
    assert manifest["release_ready"] is False
    assert manifest["can_mark_release_ready"] is False
    assert manifest["audio_copied"] is False
    assert manifest["audio_exported"] is False


def test_master_manifest_blocks_without_pair_compatibility_evidence(tmp_path: Path) -> None:
    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest()),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "pair_compatibility_evidence_missing"


def test_master_manifest_blocks_non_both_pair_compatibility_row(tmp_path: Path) -> None:
    pair_row = accepted_pair_compatibility_row()
    pair_row["score_mode"] = "minus"
    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest()),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
        pair_compatibility_jsonl=write_jsonl(tmp_path / "pair_compatibility.jsonl", [pair_row]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "pair_compatibility_not_both_mode"


def test_master_manifest_blocks_pair_path_mismatch(tmp_path: Path) -> None:
    pair_row = accepted_pair_compatibility_row()
    pair_row["instrumental_path"] = "/tmp/other-minus.wav"
    pair_row["minus_path"] = "/tmp/other-minus.wav"
    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest()),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
        pair_compatibility_jsonl=write_jsonl(tmp_path / "pair_compatibility.jsonl", [pair_row]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "pair_compatibility_no_exact_match"


def test_master_manifest_blocks_incompatible_pair_decision(tmp_path: Path) -> None:
    pair_row = accepted_pair_compatibility_row()
    pair_row["decision"] = "needs_pair_reconciliation"
    pair_row["pair_decision"] = "needs_pair_reconciliation"
    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest()),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
        pair_compatibility_jsonl=write_jsonl(tmp_path / "pair_compatibility.jsonl", [pair_row]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "pair_compatibility_not_accepted"


def test_master_manifest_blocks_pair_compatible_row_when_both_mode_has_semantic_blocker(tmp_path: Path) -> None:
    pair_row = accepted_pair_compatibility_row()
    pair_row["decision"] = "blocked_hard_case"
    pair_row["reasons"] = ["vocal_leakage_in_minus"]

    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest()),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
        pair_compatibility_jsonl=write_jsonl(tmp_path / "pair_compatibility.jsonl", [pair_row]),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "pair_compatibility_blocking_semantic_label:vocal_leakage_in_minus"


def test_master_manifest_blocks_if_either_stem_is_not_accepted(tmp_path: Path) -> None:
    vocal = accepted_vocal_manifest()
    vocal["delivery_status"] = "blocked"
    vocal["delivery_allowed"] = False
    vocal["presentation_status"] = "blocked_from_listen_pack"
    vocal["dominant_failure_reason"] = "vocal_listen_preview_not_accepted"

    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", vocal),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "source_stem_manifest_not_accepted"


def test_master_manifest_blocks_track_mismatch(tmp_path: Path) -> None:
    minus = accepted_minus_manifest("track-b")
    minus["mix_path"] = "/tmp/other-mix.wav"
    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest("track-a")),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", minus),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "track_id_mismatch"


def test_master_manifest_accepts_shared_mix_path_with_legacy_track_id_mismatch(tmp_path: Path) -> None:
    vocal = accepted_vocal_manifest("nothing-can-wait")
    minus = accepted_minus_manifest("5fc521258275")
    pair_row = accepted_pair_compatibility_row("nothing-can-wait")

    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", vocal),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", minus),
        pair_compatibility_jsonl=write_jsonl(tmp_path / "pair_compatibility.jsonl", [pair_row]),
    )

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["track_id"] == "nothing-can-wait"
    assert manifest["source_track_ids"] == {"vocal": "nothing-can-wait", "minus": "5fc521258275"}
    assert manifest["track_identity_match_method"] == "shared_mix_path"


def test_master_manifest_blocks_source_safety_violation(tmp_path: Path) -> None:
    vocal = accepted_vocal_manifest()
    vocal["safety"]["training_started"] = True

    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", vocal),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "source_stem_safety_violation"
    assert manifest["safety"]["training_started"] is True


def test_master_manifest_blocks_release_ready_source_stem(tmp_path: Path) -> None:
    vocal = accepted_vocal_manifest()
    vocal["release_status"] = "release_ready"

    manifest = build_manifest(
        vocal_manifest_json=write_json(tmp_path / "vocal_manifest.json", vocal),
        minus_manifest_json=write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest()),
    )

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "source_stem_not_listen_preview_only"


def test_master_manifest_is_promotable_by_artifact_contract(tmp_path: Path) -> None:
    output_manifest = tmp_path / "master_manifest.json"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--vocal-manifest-json",
                str(write_json(tmp_path / "vocal_manifest.json", accepted_vocal_manifest())),
                "--minus-manifest-json",
                str(write_json(tmp_path / "minus_manifest.json", accepted_minus_manifest())),
                "--pair-compatibility-jsonl",
                str(write_jsonl(tmp_path / "pair_compatibility.jsonl", [accepted_pair_compatibility_row()])),
                "--output-manifest-json",
                str(output_manifest),
                "--summary-md",
                str(summary_md),
            ]
        )
        == 0
    )
    assert can_promote_to_listen_result(output_manifest) is True
    assert "delivery_status: accepted" in summary_md.read_text(encoding="utf-8")
