from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.artifact_status_contract import ArtifactContractError, can_promote_to_listen_result, sha256_file, status_view_from_manifest
from tools.build_no_reference_listen_preview_manifest import build_manifest, is_accepted_optimizer_row, main


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def accepted_optimizer_row() -> dict:
    return {
        "track_id": "track-a",
        "candidate_id": "mode_specific_pair:vocal-a:minus-b",
        "final_decision": "mode_specific_pair_selected",
        "pair_compatibility_decision": "pair_compatible",
        "needs_pair_reconciliation": False,
        "segment_blend_needed": False,
        "mix_path": "/app/mix.wav",
        "selected_vocal_path": "/app/vocal.flac",
        "selected_minus_path": "/app/minus.flac",
        "selected_vocal_candidate_id": "vocal-a",
        "selected_minus_candidate_id": "minus-b",
        "selected_vocal_score": 0.83,
        "selected_minus_score": 0.84,
        "pair_compatibility_score": 0.76,
    }


def reconciliation_candidate_row(*, requires_dense: bool = True) -> dict:
    row = accepted_optimizer_row()
    row.update(
        {
            "candidate_id": "reconciled:vocal-a:minus-b",
            "final_decision": "pair_reconciliation_candidate",
            "requires_dense_post_reconciliation_judge": requires_dense,
            "candidate_family": "reconciled_vocal_existing_minus",
        }
    )
    return row


def dense_delivery_gate_evidence() -> dict:
    return {
        "vocal": {
            "decision": "accept_candidate",
            "score_mode": "vocal",
            "window_count": 12,
            "vocal_score": 0.91,
            "vocal_decision": "accept_candidate",
            "vocal_blockers": [],
        },
        "minus": {
            "decision": "accept_candidate",
            "score_mode": "minus",
            "window_count": 12,
            "minus_score": 0.92,
            "minus_decision": "accept_candidate",
            "minus_blockers": [],
        },
        "both": {
            "decision": "accept_candidate",
            "score_mode": "both",
            "window_count": 12,
            "pair_compatibility_score": 0.76,
            "pair_decision": "pair_compatible",
            "pair_blockers": [],
        },
    }


def legacy_generic_dense_delivery_gate_evidence() -> dict:
    return {
        "vocal": {"decision": "accept_candidate", "window_count": 12},
        "minus": {"decision": "accept_candidate", "window_count": 12},
        "both": {"decision": "accept_candidate", "window_count": 12},
    }


def dense_gate_manifest(path: Path, optimizer: Path, *, dense_delivery_gate: dict | None = None) -> Path:
    row = accepted_optimizer_row()
    payload = {
        "manifest_type": "no_reference_dense_delivery_gate",
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "dominant_failure_reason": "",
        "stem_type": "both",
        "judge_mode": "no_reference_dense_windowed",
        "source_manifest": str(optimizer),
        "source_manifest_sha256": sha256_file(optimizer),
        "accepted_count": 1,
        "skipped_count": 0,
        "accepted_listen_preview_rows": [
            {
                "track_id": row["track_id"],
                "track_hash": "track-hash-a",
                "candidate_id": row["candidate_id"],
                "mix_path": row["mix_path"],
                "vocal_path": row["selected_vocal_path"],
                "minus_path": row["selected_minus_path"],
                "instrumental_path": row["selected_minus_path"],
                "presentation_status": "accepted_for_listen_pack",
                "release_status": "listen_preview_only",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "can_mark_release_ready": False,
                "can_publish_as_release": False,
                "audio_copied": False,
                "audio_exported": False,
                "dense_delivery_gate": dense_delivery_gate or dense_delivery_gate_evidence(),
            }
        ],
        "skipped_rows": [],
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "diagnostic_audio_exported": False,
            "audio_copied": False,
            "audio_exported": False,
            "release_ready": False,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_optimizer_only_pair_is_blocked_without_dense_gate(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row()])
    manifest = build_manifest(optimizer)

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "missing_dense_delivery_gate"
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_rows"][0]["skip_reason"] == "missing_dense_delivery_gate"


def test_reconciliation_candidate_without_dense_post_flag_is_not_accepted(tmp_path: Path) -> None:
    assert is_accepted_optimizer_row(reconciliation_candidate_row(requires_dense=False)) is False
    assert is_accepted_optimizer_row(reconciliation_candidate_row(requires_dense=True)) is True
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [reconciliation_candidate_row(requires_dense=False)])
    manifest = build_manifest(optimizer)

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_rows"][0]["skip_reason"] == "not_accepted_for_listen_preview"


def test_dense_gate_manifest_becomes_listenable_manifest(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row()])
    gate = dense_gate_manifest(tmp_path / "dense_gate.json", optimizer)
    manifest = build_manifest(optimizer, dense_delivery_gate_manifest_json=gate)

    assert manifest["delivery_status"] == "accepted"
    assert manifest["delivery_allowed"] is True
    assert manifest["presentation_status"] == "accepted_for_listen_pack"
    assert manifest["release_status"] == "listen_preview_only"
    assert manifest["can_mark_release_ready"] is False
    assert manifest["safety"]["audio_copied"] is False
    assert manifest["accepted_count"] == 1
    assert manifest["dense_delivery_gate_manifest"] == str(gate)
    assert manifest["accepted_listen_preview_rows"][0]["vocal_path"] == "/app/vocal.flac"
    assert manifest["accepted_listen_preview_rows"][0]["minus_path"] == "/app/minus.flac"
    assert "dense_delivery_gate" in manifest["accepted_listen_preview_rows"][0]


def test_legacy_dense_gate_without_mode_specific_evidence_is_blocked(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row()])
    gate = dense_gate_manifest(
        tmp_path / "dense_gate.json",
        optimizer,
        dense_delivery_gate=legacy_generic_dense_delivery_gate_evidence(),
    )
    manifest = build_manifest(optimizer, dense_delivery_gate_manifest_json=gate)

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["dominant_failure_reason"] == "dense_delivery_gate_missing_mode_specific_evidence"
    assert manifest["accepted_count"] == 0
    assert manifest["skipped_rows"][0]["skip_reason"] == "dense_delivery_gate_missing_mode_specific_evidence"


def test_dense_gate_with_insufficient_embedded_windows_is_blocked(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row()])
    dense_gate = dense_delivery_gate_evidence()
    dense_gate["vocal"]["window_count"] = 1
    gate = dense_gate_manifest(tmp_path / "dense_gate.json", optimizer, dense_delivery_gate=dense_gate)
    manifest = build_manifest(optimizer, dense_delivery_gate_manifest_json=gate)

    assert manifest["delivery_status"] == "blocked"
    assert manifest["delivery_allowed"] is False
    assert manifest["dominant_failure_reason"] == "dense_delivery_gate_missing_mode_specific_evidence"
    assert "row:0:vocal_insufficient_window_count" in manifest["dense_delivery_gate_evidence_failures"]


def test_written_manifest_satisfies_artifact_status_contract(tmp_path: Path) -> None:
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [accepted_optimizer_row()])
    gate = dense_gate_manifest(tmp_path / "dense_gate.json", optimizer)
    manifest_path = tmp_path / "listen_preview_manifest.json"
    summary_path = tmp_path / "listen_preview_summary.md"

    assert main(
        [
            "--optimizer-result-jsonl",
            str(optimizer),
            "--dense-delivery-gate-manifest-json",
            str(gate),
            "--output-manifest-json",
            str(manifest_path),
            "--summary-md",
            str(summary_path),
        ]
    ) == 0

    view = status_view_from_manifest(manifest_path)
    assert view.listenable
    assert not view.deliverable
    assert can_promote_to_listen_result(manifest_path)
    assert summary_path.exists()
    assert not list(tmp_path.rglob("*.flac"))


@pytest.mark.parametrize(
    "final_decision",
    ["blocked_hard_case", "needs_segment_blend", "needs_pair_reconciliation"],
)
def test_blocked_or_plan_only_rows_are_not_promoted(tmp_path: Path, final_decision: str) -> None:
    row = accepted_optimizer_row()
    row["final_decision"] = final_decision
    if final_decision == "needs_pair_reconciliation":
        row["pair_compatibility_decision"] = "needs_pair_reconciliation"
    if final_decision == "needs_segment_blend":
        row["segment_blend_needed"] = True
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [row])
    manifest_path = tmp_path / "listen_preview_manifest.json"
    summary_path = tmp_path / "listen_preview_summary.md"

    with pytest.raises(SystemExit):
        main(
            [
                "--optimizer-result-jsonl",
                str(optimizer),
                "--output-manifest-json",
                str(manifest_path),
                "--summary-md",
                str(summary_path),
            ]
        )

    view = status_view_from_manifest(manifest_path)
    assert view.blocked
    assert not view.listenable
    with pytest.raises(ArtifactContractError):
        view.assert_promotable()


def test_missing_selected_paths_are_not_promoted(tmp_path: Path) -> None:
    row = accepted_optimizer_row()
    row["selected_minus_path"] = ""
    optimizer = write_jsonl(tmp_path / "optimizer.jsonl", [row])
    manifest = build_manifest(optimizer)

    assert manifest["delivery_allowed"] is False
    assert manifest["presentation_status"] == "blocked_from_listen_pack"
    assert manifest["accepted_count"] == 0
