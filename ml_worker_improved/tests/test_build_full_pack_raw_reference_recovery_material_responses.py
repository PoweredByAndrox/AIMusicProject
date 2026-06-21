from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_raw_reference_recovery_material_preflight import (
    build_full_pack_raw_reference_recovery_material_preflight,
)
from tools.build_full_pack_raw_reference_recovery_material_responses import (
    build_full_pack_raw_reference_recovery_material_responses,
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
    route_intent: str = "collect_clean_reference_bundle",
    next_gate: str = "collect_verified_clean_reference_bundle",
    required_actions: list[str] | None = None,
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
        "required_actions": required_actions or ["collect_clean_project_reference_export"],
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


def _bundle(
    root: Path,
    *,
    stem: str,
    verified: bool = True,
    rights_status: str = "project_owned",
    source: str = "FL Studio split mixer tracks plus full mix supplied by user",
) -> Path:
    (root / "stems").mkdir(parents=True, exist_ok=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / f"{stem}.flac").write_bytes(stem.encode("ascii"))
    _write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": f"track-{root.name}",
            "ground_truth_contract": {"verified": verified, "source": source},
            "mix": {"path": "source_mix.flac"},
            "stems": {stem: {"path": f"stems/{stem}.flac"}},
            "report_alignment_contract": {"rights_status": rights_status},
        },
    )
    return root


def test_response_builder_writes_rsp_for_clean_material_and_preflight_accepts_it(tmp_path: Path) -> None:
    exec_json = _execution_json(
        tmp_path,
        [
            _action_item("a001", stem="minus"),
            _action_item("a002", stem="guitar"),
        ],
    )
    material_root = tmp_path / "mat"
    minus_root = _bundle(material_root / "m001", stem="minus")
    guitar_root = _bundle(material_root / "m002", stem="guitar")

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[minus_root, guitar_root],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )

    assert responses["schema"] == "full_pack_raw_reference_recovery_material_responses_manifest.v1"
    assert responses["response_status"] == "ready_for_raw_reference_recovery_material_preflight"
    assert responses["action_item_count"] == 2
    assert responses["ready_response_count"] == 2
    assert responses["blocked_response_count"] == 0
    assert responses["written_response_count"] == 2
    assert responses["ready_by_stem"] == {"guitar": 1, "minus": 1}
    assert responses["delivery_allowed"] is False
    assert responses["safety"]["audio_read"] is False
    assert responses["safety"]["audio_copied"] is False
    assert responses["safety"]["ground_truth_manifest_written"] is False
    assert all(len(path.name) <= 32 for path in (tmp_path / "rrsp").iterdir() if path.is_file())

    response_rows = _read_jsonl(tmp_path / "rrsp" / "ready.jsonl")
    assert [row["response_json"] for row in response_rows] == [
        str(tmp_path / "rexec" / "rsp" / "a001.json"),
        str(tmp_path / "rexec" / "rsp" / "a002.json"),
    ]
    response_payload = _read_json(tmp_path / "rexec" / "rsp" / "a001.json")
    assert response_payload["schema"] == "full_pack_raw_reference_recovery_material_response.v1"
    assert response_payload["source_action_item_id"] == "a001"
    assert response_payload["material_root"] == str(minus_root)
    assert response_payload["accepted_as_ground_truth"] is False
    assert response_payload["delivery_allowed"] is False

    preflight = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=tmp_path / "rexec" / "rsp",
        allowed_source_roots=[material_root],
        output_dir=tmp_path / "rpre",
    )
    assert preflight["preflight_status"] == "ready_for_raw_reference_recovery_verification"
    assert preflight["ready_action_count"] == 2
    assert preflight["candidate_roots"] == [str(minus_root), str(guitar_root)]


def test_response_builder_blocks_missing_material_without_writing_rsp(tmp_path: Path) -> None:
    exec_json = _execution_json(
        tmp_path,
        [
            _action_item("a001", stem="minus"),
            _action_item("a002", stem="guitar"),
        ],
    )

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert responses["ready_response_count"] == 0
    assert responses["blocked_response_count"] == 2
    assert responses["written_response_count"] == 0
    assert not (tmp_path / "rexec" / "rsp" / "a001.json").exists()
    blocked = _read_jsonl(tmp_path / "rrsp" / "blocked.jsonl")
    assert [row["source_action_item_id"] for row in blocked] == ["a001", "a002"]
    assert all("no_matching_clean_reference_material" in row["blockers"] for row in blocked)


def test_response_builder_rejects_candidate_root_outside_explicit_allowed_roots(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem="minus")])
    outside_root = _bundle(tmp_path / "outside" / "m001", stem="minus")

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[outside_root],
        response_root=tmp_path / "rexec" / "rsp",
        allowed_source_roots=[tmp_path / "allowed"],
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert responses["ready_response_count"] == 0
    assert responses["blocked_response_count"] == 1
    assert responses["written_response_count"] == 0
    assert not (tmp_path / "rexec" / "rsp" / "a001.json").exists()
    blocked = _read_jsonl(tmp_path / "rrsp" / "blocked.jsonl")
    assert "candidate_root_outside_allowed_roots" in blocked[0]["blockers"]
    assert "no_matching_clean_reference_material" in blocked[0]["blockers"]


def test_response_builder_rejects_debug_test_artifact_material_roots(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem="guitar")])
    debug_root = _bundle(
        tmp_path / "dbg_loop" / "test_metadata_loop_builds_raw_0" / "clean" / "g001",
        stem="guitar",
    )

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[tmp_path / "dbg_loop"],
        response_root=tmp_path / "rexec" / "rsp",
        allowed_source_roots=[tmp_path],
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert responses["ready_response_count"] == 0
    assert responses["blocked_response_count"] == 1
    assert responses["written_response_count"] == 0
    assert not (tmp_path / "rexec" / "rsp" / "a001.json").exists()
    blocked = _read_jsonl(tmp_path / "rrsp" / "blocked.jsonl")
    assert "candidate_root_non_production_artifact" in blocked[0]["blockers"]
    assert "no_matching_clean_reference_material" in blocked[0]["blockers"]
    assert str(debug_root) in responses["candidate_bundle_roots"]


def test_response_builder_removes_stale_rsp_when_rerun_blocks(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem="minus")])
    material_root = tmp_path / "mat"
    minus_root = _bundle(material_root / "m001", stem="minus")
    response_root = tmp_path / "rexec" / "rsp"

    first = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[minus_root],
        response_root=response_root,
        output_dir=tmp_path / "rrsp1",
    )
    assert first["response_status"] == "ready_for_raw_reference_recovery_material_preflight"
    assert (response_root / "a001.json").exists()

    second = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[],
        response_root=response_root,
        output_dir=tmp_path / "rrsp2",
    )

    assert second["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert second["written_response_count"] == 0
    assert not (response_root / "a001.json").exists()
    preflight = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=response_root,
        allowed_source_roots=[material_root],
        output_dir=tmp_path / "rpre",
    )
    assert preflight["preflight_status"] == "blocked_waiting_for_recovery_reference_material"
    assert preflight["ready_action_count"] == 0


def test_response_builder_blocks_empty_requested_stems(tmp_path: Path) -> None:
    row = _action_item("a001", stem="minus")
    row["requested_stems"] = []
    row["target_stem_type"] = ""
    exec_json = _execution_json(tmp_path, [row])
    material_root = tmp_path / "mat"
    minus_root = _bundle(material_root / "m001", stem="minus")

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[minus_root],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert responses["written_response_count"] == 0
    blocked = _read_jsonl(tmp_path / "rrsp" / "blocked.jsonl")
    assert "requested_stems_missing" in blocked[0]["blockers"]
    assert not (tmp_path / "rexec" / "rsp" / "a001.json").exists()


def test_response_builder_discovers_nested_and_manifest_file_candidates(tmp_path: Path) -> None:
    exec_json = _execution_json(
        tmp_path,
        [
            _action_item("a001", stem="minus"),
            _action_item("a002", stem="guitar"),
        ],
    )
    material_root = tmp_path / "mat"
    minus_root = _bundle(material_root / "nested" / "deep" / "m001", stem="minus")
    guitar_root = _bundle(material_root / "m002", stem="guitar")

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[material_root / "nested", guitar_root / "ground_truth_manifest.json"],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "ready_for_raw_reference_recovery_material_preflight"
    assert responses["ready_response_count"] == 2
    assert responses["candidate_bundle_roots"] == [str(minus_root), str(guitar_root)]


def test_response_builder_rejects_unverified_or_wrong_stem_material(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem="minus")])
    material_root = tmp_path / "mat"
    _bundle(material_root / "m001", stem="minus", verified=False)
    _bundle(material_root / "m002", stem="bass")

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[material_root / "m001", material_root / "m002"],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert responses["ready_response_count"] == 0
    assert responses["blocked_response_count"] == 1
    assert responses["written_response_count"] == 0
    blocked = _read_jsonl(tmp_path / "rrsp" / "blocked.jsonl")
    assert "ground_truth_not_verified" in blocked[0]["blockers"]
    assert "candidate_missing_requested_stem:minus" in blocked[0]["blockers"]
    assert not (tmp_path / "rexec" / "rsp" / "a001.json").exists()


def test_response_builder_blocks_unsafe_execution_manifest(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem="minus")])
    payload = _read_json(exec_json)
    payload["delivery_allowed"] = True
    _write_json(exec_json, payload)

    responses = build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[tmp_path / "mat" / "m001"],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )

    assert responses["response_status"] == "blocked"
    assert responses["action_item_count"] == 0
    assert "raw_reference_recovery_execution_delivery_allowed_true" in responses["failed_checks"]
    assert _read_jsonl(tmp_path / "rrsp" / "items.jsonl") == []
