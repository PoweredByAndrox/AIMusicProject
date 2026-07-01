from __future__ import annotations

import json
from pathlib import Path

from tools.plan_mac_reference_path_recovery import build_mac_reference_path_recovery_plan, write_outputs


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mac_readiness_summary(path: Path) -> Path:
    return write_json(
        path,
        {
            "schema": "mac_candidate_readiness_summary.v1",
            "status": "blocked_source_reference_mismatch",
            "selected_candidate_id": "it7dl003",
            "source_audio_path": "/mac/source/be-nicegreenhorn.mp3",
            "source_track_identity": {
                "track_id": "be-nicegreenhorn",
                "identity_status": "actual_source_verified",
            },
            "reference_track_identity": {
                "track_id": "square-a-saw-oxygen",
                "title": "Square a Saw - Oxygen",
            },
            "reference_bundle_status": "mismatched_to_source",
            "reference_audit_status": "mismatched_to_source",
            "reference_source_audio_path": "/mac/reference/square-oxygen.flac",
            "blocker_reasons": [
                "selection_diagnostic_only",
                "full_holdout_not_evaluated",
                "reference_bundle_mismatched_to_source",
            ],
            "next_actions": [
                "run_full_holdout_resolution_for_selected_candidate",
                "attach_verified_reference_bundle_for_actual_source",
            ],
            "delivery_allowed": False,
            "listen_result_allowed": False,
        },
    )


def test_mac_readiness_summary_builds_reference_recovery_plan(tmp_path: Path) -> None:
    result = build_mac_reference_path_recovery_plan(
        mac_readiness_summary_json=mac_readiness_summary(tmp_path / "summary.json"),
        output_dir=tmp_path / "out",
        reference_search_roots=[tmp_path / "refs"],
    )

    assert result["summary"]["schema"] == "reference_path_recovery_summary.v1"
    assert result["summary"]["total_rows"] == 1
    row = result["rows"][0]
    assert row["case_id"] == "mac-it7dl003-reference"
    assert row["track_hash"] == "be-nicegreenhorn"
    assert row["stem_type"] == "minus"
    assert row["source_audio_path"] == "/mac/source/be-nicegreenhorn.mp3"
    assert row["source_track_identity"]["track_id"] == "be-nicegreenhorn"
    assert row["selected_candidate_id"] == "it7dl003"
    assert row["reference_mismatch_status"] == "reference_bundle_mismatched_to_source"
    assert row["rejected_reference_track_identity"]["track_id"] == "square-a-saw-oxygen"
    assert row["recovery_status"] == "blocked_missing_reference"
    assert row["can_recover_reference_audio"] is False
    assert row["required_next_tool"] == "manual_source_manifest_inspection"
    assert "no_audio_copy" in row["forbidden_actions"]
    assert result["manifest"]["safety"]["audio_copied"] is False


def test_mac_reference_recovery_outputs_are_metadata_only(tmp_path: Path) -> None:
    result = build_mac_reference_path_recovery_plan(
        mac_readiness_summary_json=mac_readiness_summary(tmp_path / "summary.json"),
        output_dir=tmp_path / "out",
        reference_search_roots=[tmp_path / "refs"],
    )

    write_outputs(result, tmp_path / "out")

    assert rows(tmp_path / "out" / "reference_path_recovery_plan.jsonl")[0]["case_id"] == "mac-it7dl003-reference"
    manifest = json.loads((tmp_path / "out" / "reference_path_recovery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_rows"] == 1
    assert manifest["safety"]["audio_read"] is False
    assert not list((tmp_path / "out").glob("*.flac"))
    assert not list((tmp_path / "out").glob("*.wav"))
    assert not list((tmp_path / "out").glob("*.npz"))
