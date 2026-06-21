from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_raw_reference_recovery_material_preflight import (
    build_full_pack_raw_reference_recovery_material_preflight,
)
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
        "next_gate": next_gate,
        "execution_mode": "metadata_only_external_reference_recovery",
        "requested_stems": [stem],
        "target_stem_type": stem,
        "verified_stems": [],
        "blockers": [next_gate],
        "issue_tags": [route_intent],
        "required_actions": required_actions,
        "candidate_directory": f"/app/lab_artifacts/src/{action_item_id}",
        "source_mix_candidate_path": f"/app/lab_artifacts/src/{action_item_id}/source.flac",
        "diagnostic_bundle_root": f"/app/lab_artifacts/diag/{action_item_id}",
        "diagnostic_manifest_json": f"/app/lab_artifacts/diag/{action_item_id}/gt_rej.json",
        "rights_status": "project_owned",
        "accepted_as_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def _action_plan(tmp_path: Path, rows: list[dict]) -> Path:
    rows_path = _write_jsonl(tmp_path / "ract" / "items.jsonl", rows)
    return _write_json(
        tmp_path / "ract" / "plan.json",
        {
            "schema": "full_pack_raw_reference_recovery_action_plan_manifest.v1",
            "manifest_type": "full_pack_raw_reference_recovery_action_plan",
            "action_plan_status": "ready_for_raw_reference_recovery_actions"
            if rows
            else "no_raw_reference_recovery_actions_needed",
            "failed_checks": [],
            "action_item_count": len(rows),
            "outputs": {
                "manifest_json": str(tmp_path / "ract" / "plan.json"),
                "items_jsonl": str(rows_path),
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
                "training_started": False,
            },
        },
    )


def _execution_json(tmp_path: Path, rows: list[dict]) -> Path:
    plan_json = _action_plan(tmp_path, rows)
    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=plan_json,
        output_dir=tmp_path / "rexec",
    )
    return Path(manifest["outputs"]["manifest_json"])


def _material_response(
    *,
    response_root: Path,
    material_root: Path,
    action_id: str,
    bundle_alias: str,
    stem: str,
    route_intent: str,
    required_material: str,
    verified: bool = True,
    rights_status: str = "project_owned",
) -> Path:
    bundle = material_root / action_id
    (bundle / "stems").mkdir(parents=True, exist_ok=True)
    (bundle / "source_mix.flac").write_bytes(b"mix")
    (bundle / "stems" / f"{stem}.flac").write_bytes(stem.encode("ascii"))
    manifest_json = _write_json(
        bundle / "ground_truth_manifest.json",
        {
            "track_name": f"track-{action_id}",
            "ground_truth_contract": {
                "verified": verified,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {stem: {"path": f"stems/{stem}.flac"}},
            "report_alignment_contract": {"rights_status": rights_status},
        },
    )
    return _write_json(
        response_root / f"{action_id}.json",
        {
            "schema": "full_pack_raw_reference_recovery_material_response.v1",
            "source_action_item_id": action_id,
            "source_bundle_alias": bundle_alias,
            "route_intent": route_intent,
            "required_material": required_material,
            "target_stem_type": stem,
            "requested_stems": [stem],
            "material_status": "provided_for_preflight",
            "material_root": str(bundle),
            "ground_truth_manifest_json": str(manifest_json),
            "source_mix_path": str(bundle / "source_mix.flac"),
            "target_stem_paths": {stem: str(bundle / "stems" / f"{stem}.flac")},
            "rights_status": rights_status,
            "provenance_source": "FL Studio split mixer tracks plus full mix supplied by user",
            "verification_declared": verified,
            "accepted_as_ground_truth": False,
            "delivery_allowed": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "training_started": False,
        },
    )


def test_raw_reference_recovery_material_preflight_blocks_missing_responses(tmp_path: Path) -> None:
    exec_json = _execution_json(
        tmp_path,
        [
            _action_item(
                "a001",
                stem="minus",
                route_intent="repair_or_replace_reconstruction_failed_bundle",
                next_gate="rerun_full_mix_reconstruction_audit",
                required_actions=["provide_clean_scoped_export"],
            ),
            _action_item(
                "a002",
                stem="guitar",
                route_intent="collect_clean_reference_bundle",
                next_gate="collect_verified_clean_reference_bundle",
                required_actions=["collect_clean_project_reference_export"],
            ),
            _action_item(
                "a003",
                stem="drums",
                route_intent="replace_missing_requested_stem_candidate",
                next_gate="rerun_raw_reference_bundle_worklist",
                required_actions=["collect_requested_stem_candidate"],
            ),
        ],
    )

    manifest = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=tmp_path / "rsp",
        allowed_source_roots=[tmp_path / "mat"],
        output_dir=tmp_path / "pre",
    )

    assert manifest["schema"] == "full_pack_raw_reference_recovery_material_preflight_manifest.v1"
    assert manifest["preflight_status"] == "blocked_waiting_for_recovery_reference_material"
    assert manifest["action_item_count"] == 3
    assert manifest["ready_action_count"] == 0
    assert manifest["blocked_action_count"] == 3
    assert manifest["ready_by_stem"] == {}
    assert manifest["blocked_by_stem"] == {"drums": 1, "guitar": 1, "minus": 1}
    assert manifest["candidate_roots"] == []
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["safety"]["audio_read"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["safety"]["audio_decoded"] is False
    assert manifest["safety"]["ground_truth_manifest_written"] is False

    blocked = _read_jsonl(tmp_path / "pre" / "blocked.jsonl")
    assert [row["source_action_item_id"] for row in blocked] == ["a001", "a002", "a003"]
    assert all(row["material_preflight_status"] == "blocked_missing_recovery_reference_material" for row in blocked)
    assert all("response_json_missing" in row["blockers"] for row in blocked)
    assert all(len(path.name) <= 32 for path in (tmp_path / "pre").iterdir() if path.is_file())


def test_raw_reference_recovery_material_preflight_accepts_ready_material_for_verification_only(
    tmp_path: Path,
) -> None:
    exec_json = _execution_json(
        tmp_path,
        [
            _action_item(
                "a001",
                stem="minus",
                route_intent="repair_or_replace_reconstruction_failed_bundle",
                next_gate="rerun_full_mix_reconstruction_audit",
                required_actions=["provide_clean_scoped_export"],
            ),
            _action_item(
                "a002",
                stem="guitar",
                route_intent="collect_clean_reference_bundle",
                next_gate="collect_verified_clean_reference_bundle",
                required_actions=["collect_clean_project_reference_export"],
            ),
        ],
    )
    response_root = tmp_path / "rsp"
    material_root = tmp_path / "mat"
    _material_response(
        response_root=response_root,
        material_root=material_root,
        action_id="a001",
        bundle_alias="b001",
        stem="minus",
        route_intent="repair_or_replace_reconstruction_failed_bundle",
        required_material="clean_scoped_reference_export",
    )
    _material_response(
        response_root=response_root,
        material_root=material_root,
        action_id="a002",
        bundle_alias="b002",
        stem="guitar",
        route_intent="collect_clean_reference_bundle",
        required_material="clean_project_reference_bundle",
    )

    manifest = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=response_root,
        allowed_source_roots=[material_root],
        output_dir=tmp_path / "pre",
    )

    assert manifest["preflight_status"] == "ready_for_raw_reference_recovery_verification"
    assert manifest["ready_action_count"] == 2
    assert manifest["blocked_action_count"] == 0
    assert manifest["ready_by_stem"] == {"guitar": 1, "minus": 1}
    assert manifest["candidate_roots"] == [str(material_root / "a001"), str(material_root / "a002")]
    assert manifest["delivery_allowed"] is False
    assert manifest["can_publish_as_result"] is False

    ready = _read_jsonl(tmp_path / "pre" / "ready.jsonl")
    assert [row["source_action_item_id"] for row in ready] == ["a001", "a002"]
    assert all(row["material_preflight_status"] == "ready_for_raw_reference_recovery_verification" for row in ready)
    assert all(row["accepted_as_ground_truth"] is False for row in ready)
    assert all(row["delivery_allowed"] is False for row in ready)


def test_raw_reference_recovery_material_preflight_rejects_unverified_or_outside_material(
    tmp_path: Path,
) -> None:
    exec_json = _execution_json(
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
    )
    response_root = tmp_path / "rsp"
    outside_root = tmp_path / "outside"
    _material_response(
        response_root=response_root,
        material_root=outside_root,
        action_id="a001",
        bundle_alias="b001",
        stem="minus",
        route_intent="repair_or_replace_reconstruction_failed_bundle",
        required_material="clean_scoped_reference_export",
        verified=False,
    )

    manifest = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=response_root,
        allowed_source_roots=[tmp_path / "allowed"],
        output_dir=tmp_path / "pre",
    )

    assert manifest["preflight_status"] == "blocked_waiting_for_recovery_reference_material"
    assert manifest["ready_action_count"] == 0
    assert manifest["blocked_action_count"] == 1
    blocked = _read_jsonl(tmp_path / "pre" / "blocked.jsonl")
    assert blocked[0]["material_preflight_status"] == "blocked_invalid_recovery_reference_material"
    assert "material_root_outside_allowed_roots" in blocked[0]["blockers"]
    assert "ground_truth_not_verified" in blocked[0]["blockers"]
    assert "verification_not_declared" in blocked[0]["blockers"]


def test_raw_reference_recovery_material_preflight_blocks_unsafe_execution_manifest(tmp_path: Path) -> None:
    exec_json = _execution_json(
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
    )
    payload = _read_json(exec_json)
    payload["delivery_allowed"] = True
    _write_json(exec_json, payload)

    manifest = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=tmp_path / "rsp",
        allowed_source_roots=[tmp_path / "mat"],
        output_dir=tmp_path / "pre",
    )

    assert manifest["preflight_status"] == "blocked"
    assert manifest["action_item_count"] == 0
    assert "raw_reference_recovery_execution_delivery_allowed_true" in manifest["failed_checks"]
    assert _read_jsonl(tmp_path / "pre" / "items.jsonl") == []


def test_raw_reference_recovery_material_preflight_handles_no_actions(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [])

    manifest = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=tmp_path / "rsp",
        allowed_source_roots=[tmp_path / "mat"],
        output_dir=tmp_path / "pre",
    )

    assert manifest["preflight_status"] == "no_raw_reference_recovery_material_preflight_needed"
    assert manifest["action_item_count"] == 0
    assert manifest["ready_action_count"] == 0
    assert manifest["blocked_action_count"] == 0
