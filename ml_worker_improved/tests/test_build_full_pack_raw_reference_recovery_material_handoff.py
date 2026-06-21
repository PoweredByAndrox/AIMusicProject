from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_raw_reference_recovery_material_handoff import (
    build_full_pack_raw_reference_recovery_material_handoff,
)
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


def _action_item(action_item_id: str, *, stem: str) -> dict:
    return {
        "schema": "full_pack_raw_reference_recovery_action_item.v1",
        "action_item_id": action_item_id,
        "source_queue_item_id": action_item_id.replace("a", "q", 1),
        "source_bundle_alias": action_item_id.replace("a", "b", 1),
        "source_work_item_aliases": [action_item_id.replace("a", "r", 1)],
        "route_intent": "replace_missing_requested_stem_candidate",
        "next_gate": "collect_verified_clean_reference_bundle",
        "execution_mode": "metadata_only_external_reference_recovery",
        "requested_stems": [stem],
        "target_stem_type": stem,
        "verified_stems": [],
        "blockers": ["missing_requested_stem_candidate"],
        "issue_tags": ["replace_missing_requested_stem_candidate"],
        "required_actions": ["collect_clean_project_reference_export"],
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


def _execution_json(tmp_path: Path, rows: list[dict]) -> Path:
    rows_path = _write_jsonl(tmp_path / "ract" / "items.jsonl", rows)
    plan_json = _write_json(
        tmp_path / "ract" / "plan.json",
        {
            "schema": "full_pack_raw_reference_recovery_action_plan_manifest.v1",
            "manifest_type": "full_pack_raw_reference_recovery_action_plan",
            "action_plan_status": "ready_for_raw_reference_recovery_actions" if rows else "no_raw_reference_recovery_actions_needed",
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
    manifest = execute_full_pack_raw_reference_recovery_action_plan(
        raw_reference_recovery_action_plan_json=plan_json,
        output_dir=tmp_path / "rexec",
    )
    return Path(manifest["outputs"]["manifest_json"])


def _bundle(root: Path, *, stem: str, verified: bool = True) -> Path:
    (root / "stems").mkdir(parents=True, exist_ok=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / f"{stem}.flac").write_bytes(stem.encode("ascii"))
    _write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": f"track-{root.name}",
            "ground_truth_contract": {
                "verified": verified,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {stem: {"path": f"stems/{stem}.flac"}},
            "report_alignment_contract": {"rights_status": "project_owned"},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )
    return root


def _preflight_with_material(tmp_path: Path, *, stem: str = "guitar") -> tuple[Path, Path]:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem=stem)])
    material_root = _bundle(tmp_path / "mat" / "m001", stem=stem)
    build_full_pack_raw_reference_recovery_material_responses(
        raw_reference_recovery_execution_json=exec_json,
        candidate_roots=[material_root],
        response_root=tmp_path / "rexec" / "rsp",
        output_dir=tmp_path / "rrsp",
    )
    preflight = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=tmp_path / "rexec" / "rsp",
        allowed_source_roots=[tmp_path / "mat"],
        output_dir=tmp_path / "rpre",
    )
    return Path(preflight["outputs"]["manifest_json"]), material_root


def test_recovery_material_handoff_exposes_ready_roots_for_intake_only(tmp_path: Path) -> None:
    preflight_json, material_root = _preflight_with_material(tmp_path)

    handoff = build_full_pack_raw_reference_recovery_material_handoff(
        raw_reference_recovery_material_preflight_json=preflight_json,
        output_dir=tmp_path / "rhand",
    )

    assert handoff["schema"] == "full_pack_raw_reference_recovery_material_handoff_manifest.v1"
    assert handoff["handoff_status"] == "ready_for_reference_corpus_intake"
    assert handoff["candidate_roots"] == [str(material_root)]
    assert handoff["ready_handoff_count"] == 1
    assert handoff["blocked_handoff_count"] == 0
    assert handoff["accepted_reference_rows"] == 0
    assert handoff["delivery_allowed"] is False
    assert handoff["safety"]["audio_read"] is False
    assert handoff["safety"]["audio_copied"] is False
    assert handoff["safety"]["ground_truth_manifest_written"] is False
    ready = _read_jsonl(tmp_path / "rhand" / "ready.jsonl")
    assert ready[0]["handoff_status"] == "ready_for_reference_corpus_intake"
    assert ready[0]["candidate_root"] == str(material_root)


def test_recovery_material_handoff_blocks_when_preflight_has_no_ready_rows(tmp_path: Path) -> None:
    exec_json = _execution_json(tmp_path, [_action_item("a001", stem="guitar")])
    preflight = build_full_pack_raw_reference_recovery_material_preflight(
        raw_reference_recovery_execution_json=exec_json,
        response_root=tmp_path / "rexec" / "rsp",
        allowed_source_roots=[tmp_path / "mat"],
        output_dir=tmp_path / "rpre",
    )

    handoff = build_full_pack_raw_reference_recovery_material_handoff(
        raw_reference_recovery_material_preflight_json=Path(preflight["outputs"]["manifest_json"]),
        output_dir=tmp_path / "rhand",
    )

    assert handoff["handoff_status"] == "blocked_waiting_for_recovery_reference_material"
    assert handoff["candidate_roots"] == []
    assert handoff["ready_handoff_count"] == 0
    assert handoff["blocked_handoff_count"] == 1
    assert _read_jsonl(tmp_path / "rhand" / "ready.jsonl") == []


def test_recovery_material_handoff_rejects_unsafe_ready_rows(tmp_path: Path) -> None:
    preflight_json, material_root = _preflight_with_material(tmp_path)
    ready_path = tmp_path / "rpre" / "ready.jsonl"
    rows = _read_jsonl(ready_path)
    rows[0]["delivery_allowed"] = True
    _write_jsonl(ready_path, rows)

    handoff = build_full_pack_raw_reference_recovery_material_handoff(
        raw_reference_recovery_material_preflight_json=preflight_json,
        output_dir=tmp_path / "rhand",
    )

    assert handoff["handoff_status"] == "blocked_waiting_for_recovery_reference_material"
    assert handoff["candidate_roots"] == []
    blocked = _read_jsonl(tmp_path / "rhand" / "blocked.jsonl")
    assert blocked[0]["candidate_root"] == str(material_root)
    assert "ready_row_delivery_allowed_true" in blocked[0]["blockers"]
