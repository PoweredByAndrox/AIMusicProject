from __future__ import annotations

import json
from pathlib import Path

from tools.execute_full_pack_raw_reference_recovery_action_plan import (
    execute_full_pack_raw_reference_recovery_action_plan,
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


def _action_item(
    action_item_id: str,
    *,
    stem: str,
    route_intent: str,
    next_gate: str,
    required_actions: list[str],
) -> dict:
    return {
        "schema": "full_pack_raw_reference_recovery_action_item.v1",
        "action_item_id": action_item_id,
        "source_queue_item_id": action_item_id.replace("a", "q", 1),
        "source_bundle_alias": action_item_id.replace("a", "b", 1),
        "source_work_item_aliases": [action_item_id.replace("a", "r", 1)],
        "route_intent": route_intent,
        "priority": "high",
        "next_gate": next_gate,
        "next_command_intent": f"{route_intent}_intent",
        "execution_mode": "metadata_only_external_reference_recovery",
        "requires_verified_reference_material": True,
        "docker_required_for_next_rerun": True,
        "next_docker_service": "ml-worker",
        "requested_stems": [stem],
        "target_stem_type": stem,
        "verified_stems": [],
        "blockers": [next_gate],
        "issue_tags": [route_intent],
        "required_actions": required_actions,
        "candidate_directory": f"/app/lab_artifacts/src/{action_item_id}",
        "source_mix_candidate_path": f"/app/lab_artifacts/src/{action_item_id}/source.flac",
        "source_mix_resolution": "same_directory_audio",
        "source_mix_manifest_json": "",
        "diagnostic_bundle_root": f"/app/lab_artifacts/diag/{action_item_id}",
        "diagnostic_manifest_json": f"/app/lab_artifacts/diag/{action_item_id}/gt_rej.json",
        "full_mix_reconstruction_status": "rejected",
        "failure_detail": "",
        "rights_status": "project_owned",
        "bundle_build_attempted": True,
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
        "diagnostic_context": {},
    }


def _action_plan(tmp_path: Path, rows: list[dict], **overrides: object) -> Path:
    rows_path = _write_jsonl(tmp_path / "ract" / "items.jsonl", rows)
    payload = {
        "schema": "full_pack_raw_reference_recovery_action_plan_manifest.v1",
        "manifest_type": "full_pack_raw_reference_recovery_action_plan",
        "action_plan_status": "ready_for_raw_reference_recovery_actions" if rows else "no_raw_reference_recovery_actions_needed",
        "next_action": "execute_raw_reference_recovery_actions" if rows else "none",
        "failed_checks": [],
        "action_item_count": len(rows),
        "route_intent_counts": {},
        "next_gate_counts": {},
        "blocked_by_stem": {},
        "outputs": {
            "manifest_json": str(tmp_path / "ract" / "plan.json"),
            "items_jsonl": str(rows_path),
            "progress_jsonl": str(tmp_path / "ract" / "progress.jsonl"),
            "summary_md": str(tmp_path / "ract" / "summary.md"),
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
    return _write_json(tmp_path / "ract" / "plan.json", payload)


def test_raw_reference_recovery_executor_scaffolds_metadata_requests(tmp_path: Path) -> None:
    source = _action_plan(
        tmp_path,
        [
            _action_item(
                "a001",
                stem="minus",
                route_intent="repair_or_replace_reconstruction_failed_bundle",
                next_gate="rerun_full_mix_reconstruction_audit",
                required_actions=["provide_clean_scoped_export", "rerun_full_mix_reconstruction_audit"],
            ),
            _action_item(
                "a002",
                stem="guitar",
                route_intent="collect_clean_reference_bundle",
                next_gate="collect_verified_clean_reference_bundle",
                required_actions=["collect_clean_project_reference_export", "verify_source_provenance"],
            ),
            _action_item(
                "a003",
                stem="drums",
                route_intent="replace_missing_requested_stem_candidate",
                next_gate="rerun_raw_reference_bundle_worklist",
                required_actions=["collect_requested_stem_candidate", "rerun_raw_reference_bundle_worklist"],
            ),
        ],
    )

    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=source,
        output_dir=tmp_path / "exec",
    )

    assert manifest["schema"] == "full_pack_raw_reference_recovery_action_execution_manifest.v1"
    assert manifest["execution_status"] == "blocked_waiting_for_external_reference_material"
    assert manifest["action_item_count"] == 3
    assert manifest["metadata_request_count"] == 3
    assert manifest["completed_action_count"] == 0
    assert manifest["execution_status_counts"] == {
        "blocked_waiting_for_clean_reference_bundle": 1,
        "blocked_waiting_for_clean_scoped_export": 1,
        "blocked_waiting_for_missing_requested_stem_candidate": 1,
    }
    assert manifest["route_intent_counts"] == {
        "collect_clean_reference_bundle": 1,
        "repair_or_replace_reconstruction_failed_bundle": 1,
        "replace_missing_requested_stem_candidate": 1,
    }
    assert manifest["delivery_allowed"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    rows = _read_jsonl(tmp_path / "exec" / "items.jsonl")
    by_source = {row["source_action_item_id"]: row for row in rows}
    assert by_source["a001"]["execution_status"] == "blocked_waiting_for_clean_scoped_export"
    assert by_source["a002"]["execution_status"] == "blocked_waiting_for_clean_reference_bundle"
    assert by_source["a003"]["execution_status"] == "blocked_waiting_for_missing_requested_stem_candidate"
    assert all(row["accepted_as_ground_truth"] is False for row in rows)
    assert all(row["delivery_allowed"] is False for row in rows)
    assert all(Path(row["template_json"]).is_file() for row in rows)
    assert all(len(path.name) <= 32 for path in (tmp_path / "exec").iterdir() if path.is_file())
    assert all(len(path.name) <= 32 for path in (tmp_path / "exec" / "tpl").iterdir() if path.is_file())

    template = _read_json(Path(by_source["a002"]["template_json"]))
    assert template["schema"] == "full_pack_raw_reference_recovery_execution_template.v1"
    assert template["source_action_item_id"] == "a002"
    assert template["target_stem_type"] == "guitar"
    assert template["ground_truth_manifest_written"] is False
    assert template["audio_placeholder_written"] is False


def test_raw_reference_recovery_executor_blocks_unsafe_plan(tmp_path: Path) -> None:
    source = _action_plan(
        tmp_path,
        [
            _action_item(
                "a001",
                stem="minus",
                route_intent="repair_or_replace_reconstruction_failed_bundle",
                next_gate="rerun_full_mix_reconstruction_audit",
                required_actions=["provide_clean_scoped_export"],
            )
        ],
        delivery_allowed=True,
    )

    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=source,
        output_dir=tmp_path / "exec",
    )

    assert manifest["execution_status"] == "blocked"
    assert manifest["action_item_count"] == 0
    assert "raw_reference_recovery_action_plan_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "exec" / "items.jsonl") == []


def test_raw_reference_recovery_executor_blocks_source_failed_checks(tmp_path: Path) -> None:
    source = _action_plan(
        tmp_path,
        [
            _action_item(
                "a001",
                stem="minus",
                route_intent="repair_or_replace_reconstruction_failed_bundle",
                next_gate="rerun_full_mix_reconstruction_audit",
                required_actions=["provide_clean_scoped_export"],
            )
        ],
        failed_checks=["source_plan_inconsistent"],
    )

    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=source,
        output_dir=tmp_path / "exec",
    )

    assert manifest["execution_status"] == "blocked"
    assert manifest["action_item_count"] == 0
    assert "raw_reference_recovery_action_plan_failed_checks_not_empty" in manifest["failed_checks"]
    assert manifest["source_failed_checks"] == ["source_plan_inconsistent"]
    assert _read_jsonl(tmp_path / "exec" / "blocked.jsonl") == []


def test_raw_reference_recovery_executor_blocks_unsafe_action_rows(tmp_path: Path) -> None:
    row = _action_item(
        "a001",
        stem="minus",
        route_intent="repair_or_replace_reconstruction_failed_bundle",
        next_gate="rerun_full_mix_reconstruction_audit",
        required_actions=["provide_clean_scoped_export"],
    )
    row["release_ready"] = True
    row["ground_truth_manifest_json"] = "/app/lab_artifacts/unsafe/gt.json"
    source = _action_plan(tmp_path, [row])

    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=source,
        output_dir=tmp_path / "exec",
    )

    assert manifest["execution_status"] == "blocked"
    assert manifest["action_item_count"] == 0
    assert "raw_reference_recovery_action_row_001_release_ready_true" in manifest["failed_checks"]
    assert "raw_reference_recovery_action_row_001_ground_truth_manifest_json_present" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "exec" / "items.jsonl") == []


def test_raw_reference_recovery_executor_handles_no_actions(tmp_path: Path) -> None:
    source = _action_plan(tmp_path, [])

    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=source,
        output_dir=tmp_path / "exec",
    )

    assert manifest["execution_status"] == "no_raw_reference_recovery_execution_needed"
    assert manifest["action_item_count"] == 0
    assert _read_json(tmp_path / "exec" / "exec.json")["execution_status_counts"] == {}
