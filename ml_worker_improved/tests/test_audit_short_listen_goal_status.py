from __future__ import annotations

import json
from pathlib import Path

from tools.audit_short_listen_goal_status import audit_short_listen_goal_status, main


REQUIRED_6S_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _item(alias: str, stem_type: str, audio_path: str, *, semantic: bool = False) -> dict:
    return {
        "schema": "aggregate_short_listen_pack_audio.v1",
        "source_alias": alias,
        "source_manifest": f"/app/source/{alias}/pack.json",
        "source_audio_path": f"/app/source/{alias}/{stem_type}.flac",
        "source_row": {"delivery_status": "accepted", "delivery_allowed": True, "presentation_status": "accepted_for_listen_pack"},
        "audio_path": audio_path,
        "stem_type": stem_type,
        "track_alias": "t001",
        "quality_status": "listen_preview_only_not_semantic_verified" if not semantic else "accepted",
        "semantic_quality_verified": semantic,
        "release_ready": False,
        "can_publish_as_result": False,
    }


def _pack(tmp_path: Path, *, missing_6s_stem: bool = False, release_ready: bool = False) -> tuple[Path, Path]:
    items: list[dict] = [
        _item("voc", "vocal", "audio/voc/v001.flac"),
        _item("voc", "vocal", "audio/voc/v002.flac"),
        _item("voc", "vocal", "audio/voc/v003.flac"),
        _item("noe", "minus_preview", "audio/noe/minus.flac"),
    ]
    for alias in ["noe6", "hh6", "t026"]:
        stems = REQUIRED_6S_STEMS[:-1] if missing_6s_stem and alias == "t026" else REQUIRED_6S_STEMS
        for stem in stems:
            items.append(_item(alias, stem, f"audio/{alias}/t001/{stem}.flac"))
    items.extend(
        [
            _item("mi", "minus", "audio/mi/t001/minus.flac", semantic=True),
            _item("mi", "vocal", "audio/mi/t001/vocal.flac", semantic=True),
            _item("mi", "minus", "audio/mi/t002/minus.flac", semantic=True),
            _item("mi", "vocal", "audio/mi/t002/vocal.flac", semantic=True),
        ]
    )
    groups: list[dict] = []
    for alias in ["voc", "noe", "noe6", "hh6", "t026", "mi"]:
        group_items = [item for item in items if item["source_alias"] == alias]
        groups.append(
            {
                "group_alias": alias,
                "status": "listen_preview_only",
                "item_count": len(group_items),
                "stem_types": sorted({item["stem_type"] for item in group_items}),
                "files": [item["audio_path"] for item in group_items],
                "semantic_quality_verified": all(item["semantic_quality_verified"] for item in group_items),
            }
        )

    pack = {
        "schema": "aggregate_short_listen_pack_manifest.v1",
        "pack_id": "lp_test",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": release_ready,
        "can_publish_as_result": False,
        "listenable": True,
        "deliverable": False,
        "item_count": len(items),
        "items": items,
        "failed_checks": [],
        "coverage_json": str(tmp_path / "pack" / "coverage.json"),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": "audio",
            "source_alias_strategy": "caller_supplied_validated_short_alias",
            "raw_track_or_stem_ids_in_path": False,
            "provenance_in_manifest": True,
            "metadata_filenames": ["pack.json", "coverage.json", "selected.jsonl", "progress.jsonl", "README.md"],
        },
        "contract": {
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
        },
    }
    coverage = {
        "schema": "aggregate_short_listen_pack_coverage.v1",
        "pack_id": "lp_test",
        "release_ready": False,
        "can_publish_as_result": False,
        "failed_checks": [],
        "not_claimed": ["release_ready", "publishable_result", "semantic_clean_ground_truth_for_unverified_stems"],
        "stem_groups": groups,
    }
    validation = {
        "schema": "aggregate_short_listen_pack_validation.v1",
        "validation_status": "passed",
        "failed_checks": [],
        "release_ready": False,
        "can_publish_as_result": False,
        "audio_file_count": len(items),
        "coverage_group_count": len(groups),
    }
    pack_path = _write_json(tmp_path / "pack" / "pack.json", pack)
    _write_json(tmp_path / "pack" / "coverage.json", coverage)
    validation_path = _write_json(tmp_path / "validation" / "validation.json", validation)
    return pack_path, validation_path


def _machine_gate(tmp_path: Path, *, pack: Path, validation: Path, accepted: bool = True) -> Path:
    return _write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "accepted" if accepted else "blocked",
            "final_result_allowed": accepted,
            "delivery_allowed": accepted,
            "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "pack_manifest_json": str(pack),
            "validation_json": str(validation),
            "minus_audit_json": str(tmp_path / "minus" / "audit.json"),
            "minus_audit_status": "completed" if accepted else "completed_with_blockers",
            "semantic_review_json": str(tmp_path / "semantic" / "review.json"),
            "semantic_review_evidence_count": 7 if accepted else 0,
            "failed_checks": [] if accepted else ["required_stem_quality_not_verified:drums"],
        },
    )


def _checks_by_id(audit: dict) -> dict[str, dict]:
    return {check["id"]: check for check in audit["checks"]}


def test_audit_short_listen_goal_status_keeps_goal_open_without_human_verdict(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path)

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
        machine_gate_json=_machine_gate(tmp_path, pack=pack, validation=validation),
    )

    checks = _checks_by_id(audit)
    assert audit["goal_complete"] is False
    assert audit["goal_status"] == "machine_pass_human_listen_pending"
    assert audit["goal_status_reason"] == "needs_human_listen_verdict"
    assert "human_listen_verdict_missing" in audit["remaining_blockers"]
    assert checks["short_windows_safe_names"]["status"] == "proved"
    assert checks["six_stem_coverage"]["status"] == "proved"
    assert checks["machine_acceptance_gate"]["status"] == "proved"
    assert checks["human_listen_verdict"]["status"] == "blocked"
    assert "semantic_clean_ground_truth_for_6s_stems" in audit["not_claimed"]
    assert audit["release_ready"] is False
    assert audit["can_publish_as_result"] is False


def test_audit_short_listen_goal_status_blocks_complete_pack_without_machine_gate(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path)

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
    )

    checks = _checks_by_id(audit)
    assert audit["goal_complete"] is False
    assert audit["goal_status"] == "machine_evidence_incomplete"
    assert "machine_acceptance_gate_missing" in audit["remaining_blockers"]
    assert checks["machine_acceptance_gate"]["status"] == "blocked"


def test_audit_short_listen_goal_status_counts_repair_selection_vocal_and_minus(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path)
    payload = json.loads(pack.read_text(encoding="utf-8"))
    payload["items"] = [
        item
        for item in payload["items"]
        if item["source_alias"] not in {"voc", "noe"}
    ]
    payload["items"].extend(
        [
            {
                **_item("rep", "vocal", "audio/rep/voc/v001.flac"),
                "source_row": {
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "route_intent": "vocal_candidate_replacement",
                },
            },
            {
                **_item("rep", "minus", "audio/rep/min/m001.flac"),
                "source_row": {
                    "delivery_status": "accepted",
                    "delivery_allowed": True,
                    "presentation_status": "accepted_for_listen_pack",
                    "route_intent": "minus_repair",
                },
            },
        ]
    )
    payload["item_count"] = len(payload["items"])
    pack.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
        machine_gate_json=_machine_gate(tmp_path, pack=pack, validation=validation),
    )

    checks = _checks_by_id(audit)
    assert checks["full_vocal_candidate_coverage"]["status"] == "proved"
    assert checks["minus_preview_present"]["status"] == "proved"
    assert "full_vocal_candidate_count_too_small" not in audit["remaining_blockers"]
    assert "minus_preview_missing" not in audit["remaining_blockers"]
    assert audit["goal_status"] == "machine_pass_human_listen_pending"


def test_audit_short_listen_goal_status_blocks_incomplete_6s_stem_coverage(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path, missing_6s_stem=True)

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
        machine_gate_json=_machine_gate(tmp_path, pack=pack, validation=validation),
    )

    checks = _checks_by_id(audit)
    assert checks["six_stem_coverage"]["status"] == "not_proved"
    assert "six_stem_coverage_incomplete" in audit["remaining_blockers"]


def test_audit_short_listen_goal_status_cli_fails_machine_incomplete_pack(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path, missing_6s_stem=True)

    exit_code = main(
        [
            "--pack-manifest-json",
            str(pack),
            "--validation-json",
            str(validation),
            "--output-dir",
            str(tmp_path / "audit_cli"),
        ]
    )

    assert exit_code == 2


def test_audit_short_listen_goal_status_blocks_failed_validation(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path)
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["validation_status"] = "failed"
    payload["failed_checks"] = ["audio_sha256_mismatch"]
    validation.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
        machine_gate_json=_machine_gate(tmp_path, pack=pack, validation=validation),
    )

    checks = _checks_by_id(audit)
    assert checks["aggregate_pack_validation"]["status"] == "blocked"
    assert "aggregate_pack_validation_failed" in audit["remaining_blockers"]


def test_audit_short_listen_goal_status_refuses_release_ready_pack(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path, release_ready=True)

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
    )

    checks = _checks_by_id(audit)
    assert checks["safety_contract"]["status"] == "blocked"
    assert "release_ready_true" in audit["remaining_blockers"]


def test_audit_short_listen_goal_status_reports_rejected_human_feedback(tmp_path: Path) -> None:
    pack, validation = _pack(tmp_path)
    feedback = _write_json(
        tmp_path / "feedback" / "listen_feedback.json",
        {
            "schema": "short_listen_feedback_manifest.v1",
            "overall_listen_verdict": "rejected",
            "failed_checks": [],
            "items": [
                {
                    "audio_path": "audio/noe/minus.flac",
                    "verdict": "rejected",
                    "issue_tags": ["muffled_minus", "residual_vocal_middle"],
                }
            ],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )

    audit = audit_short_listen_goal_status(
        pack_manifest_json=pack,
        validation_json=validation,
        output_dir=tmp_path / "audit",
        human_listen_verdict_json=feedback,
        machine_gate_json=_machine_gate(tmp_path, pack=pack, validation=validation),
    )

    checks = _checks_by_id(audit)
    assert audit["goal_complete"] is False
    assert audit["goal_status"] == "human_listen_rejected"
    assert "human_listen_verdict_rejected" in audit["remaining_blockers"]
    assert checks["human_listen_verdict"]["evidence"]["rejected_audio_paths"] == ["audio/noe/minus.flac"]
    assert audit["human_feedback"]["overall_listen_verdict"] == "rejected"
    assert audit["human_feedback"]["rejected_items"] == [
        {
            "audio_path": "audio/noe/minus.flac",
            "issue_tags": ["muffled_minus", "residual_vocal_middle"],
            "note": "",
            "source_alias": "",
            "stem_type": "",
            "track_alias": "",
        }
    ]
    assert audit["human_feedback"]["issue_tag_counts"] == {
        "muffled_minus": 1,
        "residual_vocal_middle": 1,
    }
