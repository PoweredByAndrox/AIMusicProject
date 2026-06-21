from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_raw_reference_recovery_action_plan import (
    build_full_pack_raw_reference_recovery_action_plan,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _recovery_row(
    queue_item_id: str,
    *,
    stem: str,
    route_intent: str,
    required_actions: list[str],
    blockers: list[str],
) -> dict:
    return {
        "schema": "full_pack_raw_reference_bundle_recovery_item.v1",
        "queue_item_id": queue_item_id,
        "source_bundle_alias": queue_item_id.replace("q", "b", 1),
        "source_verified_bundle_status": "blocked",
        "route_intent": route_intent,
        "priority": "high",
        "requested_stems": [stem],
        "target_stem_type": stem,
        "verified_stems": [],
        "blockers": blockers,
        "issue_tags": blockers,
        "required_actions": required_actions,
        "source_work_item_aliases": [queue_item_id.replace("q", "r", 1)],
        "source_work_item_count": 1,
        "candidate_directory": f"/app/lab_artifacts/src/{queue_item_id}",
        "source_mix_candidate_path": f"/app/lab_artifacts/src/{queue_item_id}/source.flac",
        "source_mix_resolution": "same_directory_audio",
        "source_mix_manifest_json": "",
        "diagnostic_bundle_root": f"/app/lab_artifacts/diag/{queue_item_id}",
        "diagnostic_manifest_json": f"/app/lab_artifacts/diag/{queue_item_id}/gt_rej.json",
        "ground_truth_manifest_json": "",
        "candidate_root": "",
        "full_mix_reconstruction_status": "rejected",
        "failure_detail": "",
        "rights_status": "project_owned",
        "bundle_build_attempted": True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "diagnostic_context": {"source_blockers": blockers},
    }


def _recovery_queue(tmp_path: Path, rows: list[dict], **overrides: object) -> Path:
    rows_path = _write_jsonl(tmp_path / "rrq" / "rows.jsonl", rows)
    payload = {
        "schema": "full_pack_raw_reference_bundle_recovery_queue_manifest.v1",
        "manifest_type": "full_pack_raw_reference_bundle_recovery_queue",
        "queue_status": "ready_for_raw_reference_recovery" if rows else "no_raw_reference_recovery_needed",
        "failed_checks": [],
        "queue_item_count": len(rows),
        "route_intent_counts": {},
        "blocked_by_stem": {},
        "outputs": {
            "manifest_json": str(tmp_path / "rrq" / "queue.json"),
            "rows_jsonl": str(rows_path),
            "progress_jsonl": str(tmp_path / "rrq" / "progress.jsonl"),
            "summary_md": str(tmp_path / "rrq" / "summary.md"),
        },
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "ground_truth_manifest_written": False,
            "accepted_rows_not_imported": True,
            "training_started": False,
        },
    }
    payload.update(overrides)
    return _write_json(tmp_path / "rrq" / "queue.json", payload)


def test_raw_reference_recovery_action_plan_maps_recovery_routes(tmp_path: Path) -> None:
    source = _recovery_queue(
        tmp_path,
        [
            _recovery_row(
                "q001",
                stem="minus",
                route_intent="repair_or_replace_reconstruction_failed_bundle",
                required_actions=["provide_clean_scoped_export", "rerun_raw_reference_verified_bundle_bridge"],
                blockers=["full_mix_reconstruction_not_accepted"],
            ),
            _recovery_row(
                "q002",
                stem="guitar",
                route_intent="collect_clean_reference_bundle",
                required_actions=["collect_clean_project_reference_export", "verify_source_provenance"],
                blockers=["source_mix_from_listen_pack_manifest_requires_clean_reference_bundle"],
            ),
            _recovery_row(
                "q003",
                stem="drums",
                route_intent="replace_missing_requested_stem_candidate",
                required_actions=["collect_requested_stem_candidate", "rerun_raw_reference_bundle_worklist"],
                blockers=["verified_bundle_missing_requested_stem:drums"],
            ),
        ],
    )

    manifest = build_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_queue_json=source,
        output_dir=tmp_path / "plan",
    )

    assert manifest["schema"] == "full_pack_raw_reference_recovery_action_plan_manifest.v1"
    assert manifest["action_plan_status"] == "ready_for_raw_reference_recovery_actions"
    assert manifest["action_item_count"] == 3
    assert manifest["route_intent_counts"] == {
        "collect_clean_reference_bundle": 1,
        "repair_or_replace_reconstruction_failed_bundle": 1,
        "replace_missing_requested_stem_candidate": 1,
    }
    assert manifest["blocked_by_stem"] == {"drums": 1, "guitar": 1, "minus": 1}
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "plan" / "items.jsonl")
    by_source = {row["source_queue_item_id"]: row for row in rows}
    assert by_source["q001"]["next_gate"] == "rerun_full_mix_reconstruction_audit"
    assert by_source["q002"]["next_gate"] == "collect_verified_clean_reference_bundle"
    assert by_source["q003"]["next_gate"] == "rerun_raw_reference_bundle_worklist"
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(row["accepted_as_ground_truth"] is False for row in rows)
    assert all(len(path.name) <= 32 for path in (tmp_path / "plan").iterdir() if path.is_file())


def test_raw_reference_recovery_action_plan_blocks_unsafe_queue(tmp_path: Path) -> None:
    source = _recovery_queue(
        tmp_path,
        [_recovery_row(
            "q001",
            stem="minus",
            route_intent="repair_or_replace_reconstruction_failed_bundle",
            required_actions=["provide_clean_scoped_export"],
            blockers=["full_mix_reconstruction_not_accepted"],
        )],
        delivery_allowed=True,
    )

    manifest = build_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_queue_json=source,
        output_dir=tmp_path / "plan",
    )

    assert manifest["action_plan_status"] == "blocked"
    assert manifest["action_item_count"] == 0
    assert "raw_reference_recovery_queue_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "plan" / "items.jsonl") == []


def test_raw_reference_recovery_action_plan_blocks_unsafe_rows(tmp_path: Path) -> None:
    row = _recovery_row(
        "q001",
        stem="minus",
        route_intent="repair_or_replace_reconstruction_failed_bundle",
        required_actions=["provide_clean_scoped_export"],
        blockers=["full_mix_reconstruction_not_accepted"],
    )
    row["release_ready"] = True
    row["candidate_root"] = "/app/lab_artifacts/unsafe/candidate"
    source = _recovery_queue(tmp_path, [row])

    manifest = build_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_queue_json=source,
        output_dir=tmp_path / "plan",
    )

    assert manifest["action_plan_status"] == "blocked"
    assert manifest["action_item_count"] == 0
    assert "raw_reference_recovery_row_001_release_ready_true" in manifest["failed_checks"]
    assert "raw_reference_recovery_row_001_candidate_root_present" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "plan" / "items.jsonl") == []


def test_raw_reference_recovery_action_plan_handles_no_recovery_rows(tmp_path: Path) -> None:
    source = _recovery_queue(tmp_path, [])

    manifest = build_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_queue_json=source,
        output_dir=tmp_path / "plan",
    )

    assert manifest["action_plan_status"] == "no_raw_reference_recovery_actions_needed"
    assert manifest["action_item_count"] == 0
    assert _read_json(tmp_path / "plan" / "plan.json")["route_intent_counts"] == {}
