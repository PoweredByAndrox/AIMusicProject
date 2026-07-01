from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import summarize_mac_candidate_readiness as readiness


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_readiness_blocks_pitch_guard_failed_diagnostic_manifest(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "cmp" / "manifest.json",
        {
            "schema": "mac_mel_residual_chunk_composition.v1",
            "status": "composition_blocked_pitch_guard",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "chunk_composition_diagnostic_only",
            "failed_checks": ["pitch_guard_failed"],
            "before_score": {"total_fail_regions": 46},
            "after_score": {"total_fail_regions": 7, "verdict": "fail"},
        },
    )

    summary = readiness.summarize_manifest(manifest)

    assert summary["status"] == "blocked_failed_objective_gate"
    assert summary["listen_result_allowed"] is False
    assert summary["delivery_allowed"] is False
    assert summary["objective_summary"] == {
        "before_fail_regions": 46,
        "after_fail_regions": 7,
        "region_delta": 39,
        "pitch_guard_verdict": "fail",
    }
    assert "pitch_guard_failed" in summary["blocker_reasons"]
    assert "diagnostic_presentation_status" in summary["blocker_reasons"]
    assert "exact_reference_audit_missing" in summary["blocker_reasons"]
    assert summary["next_actions"][0] == "continue_model_routing_or_training_until_objective_gate_passes"


def test_readiness_marks_pareto_selection_as_full_holdout_not_listen_result(tmp_path: Path) -> None:
    selection = _write_json(
        tmp_path / "sel" / "selection.json",
        {
            "schema": "mac_residual_pareto_selection.v1",
            "status": "pareto_candidate_selected_for_full_holdout",
            "selected_candidate_id": "it7dl003",
            "selected_candidate": {
                "candidate_id": "it7dl003",
                "schema": "mac_residual_damage_limited_repair_route.v1",
                "status": "damage_limited_repair_passed",
                "presentation_status": "damage_limited_repair_diagnostic_only",
                "full_holdout_resolution_status": "not_evaluated",
                "pitch_pass": True,
                "damage_pass": True,
                "eligible_for_promotion": True,
                "promotion_blocker_reasons": [],
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "pareto_selection_diagnostic_only",
        },
    )

    summary = readiness.summarize_manifest(selection)

    assert summary["status"] == "blocked_needs_full_holdout_and_reference"
    assert summary["selected_candidate_id"] == "it7dl003"
    assert summary["listen_result_allowed"] is False
    assert summary["candidate_ready_for_full_holdout"] is True
    assert "selection_diagnostic_only" in summary["blocker_reasons"]
    assert "full_holdout_not_evaluated" in summary["blocker_reasons"]
    assert "exact_reference_audit_missing" in summary["blocker_reasons"]
    assert "run_full_holdout_resolution_for_selected_candidate" in summary["next_actions"]


def test_readiness_uses_full_holdout_eligibility_separate_from_promotion(tmp_path: Path) -> None:
    selection = _write_json(
        tmp_path / "sel" / "selection.json",
        {
            "schema": "mac_residual_pareto_selection.v1",
            "status": "pareto_candidate_selected_for_full_holdout",
            "selected_candidate_id": "it7dl003",
            "selected_candidate": {
                "candidate_id": "it7dl003",
                "schema": "mac_residual_damage_limited_repair_route.v1",
                "status": "damage_limited_repair_passed",
                "presentation_status": "damage_limited_repair_diagnostic_only",
                "full_holdout_resolution_status": "not_evaluated",
                "pitch_pass": True,
                "damage_pass": True,
                "eligible_for_full_holdout": True,
                "eligible_for_delivery": False,
                "eligible_for_promotion": False,
                "delivery_blocker_reasons": [
                    "full_holdout_not_passed",
                    "reference_bundle_not_verified_for_actual_source",
                ],
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "pareto_selection_diagnostic_only",
        },
    )

    summary = readiness.summarize_manifest(selection)

    assert summary["status"] == "blocked_needs_full_holdout_and_reference"
    assert summary["candidate_ready_for_full_holdout"] is True
    assert summary["listen_result_allowed"] is False
    assert "run_full_holdout_resolution_for_selected_candidate" in summary["next_actions"]


def test_readiness_uses_passed_full_holdout_validation_for_selected_candidate(tmp_path: Path) -> None:
    selection = _write_json(
        tmp_path / "sel" / "selection.json",
        {
            "schema": "mac_residual_pareto_selection.v1",
            "status": "pareto_candidate_selected_for_full_holdout",
            "selected_candidate_id": "it7dl003",
            "selected_candidate": {
                "candidate_id": "it7dl003",
                "manifest": "/mac/it7dl003/r001/manifest.json",
                "pitch_pass": True,
                "damage_pass": True,
                "eligible_for_full_holdout": True,
                "full_holdout_resolution_status": "not_evaluated",
                "presentation_status": "damage_limited_repair_diagnostic_only",
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "pareto_selection_diagnostic_only",
        },
    )
    holdout = _write_json(
        tmp_path / "holdout" / "manifest.json",
        {
            "schema": "mac_residual_candidate_full_holdout_validation.v1",
            "status": "validation_passed",
            "source_candidate_manifest": "/mac/it7dl003/r001/manifest.json",
            "full_holdout_resolution_status": "passed",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "residual_candidate_full_holdout_diagnostic_only",
        },
    )

    summary = readiness.summarize_manifest(selection, full_holdout_validation_path=holdout)

    assert summary["status"] == "blocked_missing_reference"
    assert summary["candidate_ready_for_full_holdout"] is True
    assert summary["full_holdout_validation_status"] == "validation_passed"
    assert "full_holdout_not_evaluated" not in summary["blocker_reasons"]
    assert "run_full_holdout_resolution_for_selected_candidate" not in summary["next_actions"]
    assert "exact_reference_audit_missing" in summary["blocker_reasons"]
    assert summary["listen_result_allowed"] is False


def test_readiness_blocks_failed_full_holdout_validation_for_selected_candidate(tmp_path: Path) -> None:
    selection = _write_json(
        tmp_path / "sel" / "selection.json",
        {
            "schema": "mac_residual_pareto_selection.v1",
            "status": "pareto_candidate_selected_for_full_holdout",
            "selected_candidate_id": "it7dl003",
            "selected_candidate": {
                "candidate_id": "it7dl003",
                "manifest": "/mac/it7dl003/r001/manifest.json",
                "pitch_pass": True,
                "damage_pass": True,
                "eligible_for_full_holdout": True,
                "full_holdout_resolution_status": "not_evaluated",
                "presentation_status": "damage_limited_repair_diagnostic_only",
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "pareto_selection_diagnostic_only",
        },
    )
    holdout = _write_json(
        tmp_path / "holdout" / "manifest.json",
        {
            "schema": "mac_residual_candidate_full_holdout_validation.v1",
            "status": "validation_blocked_no_holdout_evidence",
            "source_candidate_manifest": "/mac/it7dl003/r001/manifest.json",
            "full_holdout_resolution_status": "failed",
            "failed_checks": ["no_holdout_evidence"],
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "residual_candidate_full_holdout_diagnostic_only",
        },
    )

    summary = readiness.summarize_manifest(selection, full_holdout_validation_path=holdout)

    assert summary["status"] == "blocked_failed_full_holdout_validation"
    assert summary["full_holdout_validation_status"] == "validation_blocked_no_holdout_evidence"
    assert "full_holdout_validation_failed" in summary["blocker_reasons"]
    assert "rerun_or_attach_independent_full_holdout_evidence" in summary["next_actions"]
    assert summary["listen_result_allowed"] is False


def test_readiness_blocks_unlinked_full_holdout_validation_for_selected_candidate(tmp_path: Path) -> None:
    selection = _write_json(
        tmp_path / "sel" / "selection.json",
        {
            "schema": "mac_residual_pareto_selection.v1",
            "status": "pareto_candidate_selected_for_full_holdout",
            "selected_candidate_id": "it7dl003",
            "selected_candidate": {
                "candidate_id": "it7dl003",
                "manifest": "/mac/it7dl003/r001/manifest.json",
                "pitch_pass": True,
                "damage_pass": True,
                "eligible_for_full_holdout": True,
                "full_holdout_resolution_status": "not_evaluated",
                "presentation_status": "damage_limited_repair_diagnostic_only",
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "pareto_selection_diagnostic_only",
        },
    )
    holdout = _write_json(
        tmp_path / "holdout" / "manifest.json",
        {
            "schema": "mac_residual_candidate_full_holdout_validation.v1",
            "status": "validation_passed",
            "source_candidate_manifest": "/mac/other/r001/manifest.json",
            "full_holdout_resolution_status": "passed",
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "residual_candidate_full_holdout_diagnostic_only",
        },
    )

    summary = readiness.summarize_manifest(selection, full_holdout_validation_path=holdout)

    assert summary["status"] == "blocked_failed_full_holdout_validation"
    assert "full_holdout_validation_not_linked_to_selected_candidate" in summary["blocker_reasons"]
    assert "rerun_or_attach_independent_full_holdout_evidence" in summary["next_actions"]
    assert summary["listen_result_allowed"] is False


def test_readiness_blocks_mismatched_reference_bundle_for_actual_source(tmp_path: Path) -> None:
    selection = _write_json(
        tmp_path / "sel" / "selection.json",
        {
            "schema": "mac_residual_pareto_selection.v1",
            "status": "pareto_candidate_selected_for_full_holdout",
            "selected_candidate_id": "it7dl003",
            "selected_candidate": {
                "candidate_id": "it7dl003",
                "source_audio_path": "/mac/source/be-nicegreenhorn.mp3",
                "source_track_identity": {
                    "track_id": "be-nicegreenhorn",
                    "identity_status": "actual_source_verified",
                },
                "pitch_pass": True,
                "damage_pass": True,
                "eligible_for_promotion": True,
                "full_holdout_resolution_status": "not_evaluated",
                "presentation_status": "damage_limited_repair_diagnostic_only",
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "pareto_selection_diagnostic_only",
        },
    )
    reference = _write_json(
        tmp_path / "gt" / "ground_truth_manifest.json",
        {
            "schema": "mac_reference_bundle.v1",
            "track_name": "Square a Saw - Oxygen",
            "source_audio_path": "/mac/source/square-oxygen.flac",
            "sample_rate": 44100,
            "duration_sec": 259.556,
        },
    )

    summary = readiness.summarize_manifest(selection, reference_audit_path=reference)

    assert summary["status"] == "blocked_source_reference_mismatch"
    assert summary["reference_audit_status"] == "mismatched_to_source"
    assert summary["source_track_identity"]["track_id"] == "be-nicegreenhorn"
    assert summary["reference_track_identity"]["track_id"] == "square-a-saw-oxygen"
    assert "reference_bundle_mismatched_to_source" in summary["blocker_reasons"]
    assert "attach_verified_reference_bundle_for_actual_source" in summary["next_actions"]
    assert summary["listen_result_allowed"] is False


def test_cli_writes_readiness_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "accepted" / "pack.json",
        {
            "schema": "fully_automatic_stem_listen_pack.v1",
            "status": "accepted",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "listen_pack_created": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    out = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_mac_candidate_readiness.py",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
    )

    assert readiness.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "accepted_for_listen_pack"
    assert summary["listen_result_allowed"] is True
    assert summary["blocker_reasons"] == []


def test_readiness_blocks_accepted_manifest_when_audibility_audit_fails(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "accepted" / "pack.json",
        {
            "schema": "fully_automatic_stem_listen_pack.v1",
            "status": "accepted",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "listen_pack_created": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )
    audibility = _write_json(
        tmp_path / "aud" / "manifest.json",
        {
            "schema": "mac_candidate_audibility_delta.v1",
            "status": "audibility_delta_blocked",
            "verdict": "fail",
            "blocker_reasons": ["audibility_delta_below_gate"],
            "metrics": {
                "diff_rel_db_vs_source": -78.0,
                "active_changed_sec": 0.0,
                "max_window_diff_ratio": 0.001,
            },
            "delivery_allowed": False,
            "listen_pack_created": False,
            "presentation_status": "audibility_delta_diagnostic_only",
        },
    )

    summary = readiness.summarize_manifest(manifest, audibility_audit_path=audibility)

    assert summary["status"] == "blocked_inaudible_candidate"
    assert summary["listen_result_allowed"] is False
    assert summary["delivery_allowed"] is False
    assert summary["audibility_audit_status"] == "fail"
    assert "audibility_delta_below_gate" in summary["blocker_reasons"]
    assert "run_full_region_repair_or_reject_inaudible_candidate" in summary["next_actions"]


def test_readiness_can_require_audibility_audit_before_acceptance(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "accepted" / "pack.json",
        {
            "schema": "fully_automatic_stem_listen_pack.v1",
            "status": "accepted",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "listen_pack_created": True,
            "presentation_status": "accepted_for_listen_pack",
        },
    )

    summary = readiness.summarize_manifest(manifest, require_audibility_audit=True)

    assert summary["status"] == "blocked_missing_audibility_audit"
    assert summary["listen_result_allowed"] is False
    assert summary["delivery_allowed"] is False
    assert summary["audibility_audit_status"] == "missing"
    assert "audibility_audit_missing" in summary["blocker_reasons"]
    assert "run_audibility_delta_audit_for_candidate" in summary["next_actions"]
