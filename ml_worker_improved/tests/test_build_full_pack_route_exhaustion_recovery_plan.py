from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_pack_route_exhaustion_recovery_plan import build_full_pack_route_exhaustion_recovery_plan


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _machine_gate(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "failed_checks": [
                "minus_audit_blocked_minus_present",
                "minus_audit_warnings_present",
                "required_stem_train_ready_missing:minus",
                "required_stem_train_ready_missing:drums",
                "required_stem_train_ready_missing:bass",
                "required_stem_train_ready_missing:other",
                "required_stem_train_ready_missing:guitar",
                "required_stem_train_ready_missing:piano",
            ],
            "required_stems": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
            "stem_types": ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"],
            "missing_required_stems": [],
            "unverified_required_stems": [],
            "train_ready_stems": ["vocal"],
            "release_ready": False,
        },
    )


def _minus_audit(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "minqa" / "audit.json",
        {
            "schema": "short_minus_quality_audit.v1",
            "audit_status": "completed_with_blockers",
            "delivery_allowed": False,
            "blocked_minus_count": 1,
            "warning_count": 1,
            "failed_checks": [
                "minus_audit_blocked:audio/stm/t001/minus.flac:severe_vocal_bleed_in_minus",
                "minus_audit_blocked:audio/stm/t001/minus.flac:windowed_minus_not_accepted",
            ],
            "release_ready": False,
        },
    )


def _route_expansion(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "exp" / "minus_route_expansion_manifest.json",
        {
            "schema_version": 1,
            "profile": "hard_case",
            "delivery_allowed": False,
            "release_ready": False,
            "summary": {
                "accepted_minus_candidates": 0,
                "minus_candidates_generated": 4,
                "jobs_attempted": 3,
                "jobs_failed": 0,
                "route_space_exhausted": True,
                "route_space_limited": False,
                "severe_vocal_bleed_in_minus_count": 3,
                "severe_phase_collapse_count": 1,
                "training_started": False,
                "thresholds_relaxed": False,
                "selected_route_ids": ["htdemucs_6s", "htdemucs_ft"],
                "rejected_route_ids": ["htdemucs_6s", "htdemucs_ft"],
            },
        },
    )


def _reference_gap(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "delivery_allowed": False,
            "gap_count": 6,
            "missing_reference_stem_count": 13,
            "missing_training_job_count": 4,
            "release_ready": False,
        },
    )


def _recovery_handoff(tmp_path: Path, *, ready_count: int = 0) -> Path:
    return _write_json(
        tmp_path / "rhand" / "handoff.json",
        {
            "schema": "full_pack_raw_reference_recovery_material_handoff_manifest.v1",
            "handoff_status": "ready_for_reference_corpus_intake"
            if ready_count
            else "blocked_waiting_for_recovery_reference_material",
            "delivery_allowed": False,
            "ready_handoff_count": ready_count,
            "blocked_handoff_count": 10 if not ready_count else 0,
            "candidate_roots": [str(tmp_path / "refs" / "r001")] if ready_count else [],
            "release_ready": False,
        },
    )


def _reference_intake_after_duplicate_rejection(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "int" / "intake.json",
        {
            "schema": "full_pack_reference_corpus_intake_manifest.v1",
            "intake_status": "no_accepted_references",
            "delivery_allowed": False,
            "release_ready": False,
            "candidate_roots": [str(tmp_path / "refs" / "r001")],
            "accepted_reference_rows": 0,
            "rejected_reference_rows": 5,
            "remaining_reference_stem_count": 13,
            "rejection_reason_counts": {"duplicate_existing_track_stem": 5},
        },
    )


def test_route_exhaustion_plan_escalates_to_reference_and_new_route_actions(tmp_path: Path) -> None:
    manifest = build_full_pack_route_exhaustion_recovery_plan(
        machine_gate_json=_machine_gate(tmp_path),
        minus_audit_json=_minus_audit(tmp_path),
        minus_route_expansion_json=_route_expansion(tmp_path),
        reference_gap_plan_json=_reference_gap(tmp_path),
        raw_reference_handoff_json=_recovery_handoff(tmp_path),
        output_dir=tmp_path / "out",
        command_args=["--minus-route-expansion-json", "exp.json"],
    )

    assert manifest["schema"] == "full_pack_route_exhaustion_recovery_plan.v1"
    assert manifest["plan_status"] == "blocked_waiting_for_reference_or_new_route_material"
    assert manifest["route_space_exhausted"] is True
    assert manifest["full_pack_required_stems"] == ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]
    assert manifest["full_pack_present_stems"] == ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]
    assert manifest["full_pack_missing_required_stems"] == []
    assert manifest["full_pack_train_ready_stems"] == ["vocal"]
    assert manifest["full_pack_missing_train_ready_stems"] == [
        "minus",
        "drums",
        "bass",
        "other",
        "guitar",
        "piano",
    ]
    assert manifest["accepted_minus_candidates"] == 0
    assert manifest["action_count"] == 3
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["not_claimed"] == ["accepted_stems", "listen_result", "audio_repaired", "model_promoted"]

    actions = _read_jsonl(Path(manifest["actions_jsonl"]))
    assert [row["action_type"] for row in actions] == [
        "collect_verified_reference_material",
        "design_new_minus_route_or_training_candidate",
        "rerun_full_pack_gates_after_new_candidate",
    ]
    assert actions[0]["target_stem_type"] == "minus"
    assert actions[0]["blocking_evidence"]["full_pack_required_stems"] == manifest["full_pack_required_stems"]
    assert actions[0]["blocking_evidence"]["full_pack_missing_train_ready_stems"] == [
        "minus",
        "drums",
        "bass",
        "other",
        "guitar",
        "piano",
    ]
    assert actions[0]["blocking_evidence"]["ready_raw_reference_handoff_count"] == 0
    assert actions[1]["blocking_evidence"]["route_space_exhausted"] is True
    assert all(row["delivery_allowed"] is False for row in actions)
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())


def test_route_exhaustion_plan_uses_ready_handoff_before_new_route_design(tmp_path: Path) -> None:
    manifest = build_full_pack_route_exhaustion_recovery_plan(
        machine_gate_json=_machine_gate(tmp_path),
        minus_audit_json=_minus_audit(tmp_path),
        minus_route_expansion_json=_route_expansion(tmp_path),
        reference_gap_plan_json=_reference_gap(tmp_path),
        raw_reference_handoff_json=_recovery_handoff(tmp_path, ready_count=1),
        output_dir=tmp_path / "out",
    )

    actions = _read_jsonl(Path(manifest["actions_jsonl"]))
    assert manifest["plan_status"] == "ready_for_reference_material_intake"
    assert [row["action_type"] for row in actions] == [
        "ingest_ready_reference_material",
        "rerun_full_pack_gates_after_new_candidate",
    ]
    assert actions[0]["candidate_roots"] == [str(tmp_path / "refs" / "r001")]


def test_route_exhaustion_plan_does_not_reingest_duplicate_only_ready_handoff(tmp_path: Path) -> None:
    manifest = build_full_pack_route_exhaustion_recovery_plan(
        machine_gate_json=_machine_gate(tmp_path),
        minus_audit_json=_minus_audit(tmp_path),
        minus_route_expansion_json=_route_expansion(tmp_path),
        reference_gap_plan_json=_reference_gap(tmp_path),
        raw_reference_handoff_json=_recovery_handoff(tmp_path, ready_count=1),
        reference_intake_json=_reference_intake_after_duplicate_rejection(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["plan_status"] == "blocked_waiting_for_reference_or_new_route_material"
    assert manifest["reference_intake_accepted_reference_rows"] == 0
    assert manifest["reference_intake_rejected_reference_rows"] == 5
    assert manifest["reference_intake_candidate_roots_consumed"] is True
    actions = _read_jsonl(Path(manifest["actions_jsonl"]))
    assert [row["action_type"] for row in actions] == [
        "collect_verified_reference_material",
        "design_new_minus_route_or_training_candidate",
        "rerun_full_pack_gates_after_new_candidate",
    ]
    assert all(row["candidate_roots"] == [] for row in actions)


def test_route_exhaustion_plan_blocks_when_route_space_not_exhausted(tmp_path: Path) -> None:
    route_expansion = _route_expansion(tmp_path)
    payload = json.loads(route_expansion.read_text(encoding="utf-8"))
    payload["summary"]["route_space_exhausted"] = False
    route_expansion.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_full_pack_route_exhaustion_recovery_plan(
        machine_gate_json=_machine_gate(tmp_path),
        minus_audit_json=_minus_audit(tmp_path),
        minus_route_expansion_json=route_expansion,
        reference_gap_plan_json=_reference_gap(tmp_path),
        raw_reference_handoff_json=_recovery_handoff(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["plan_status"] == "blocked_route_space_not_exhausted"
    assert "minus_route_space_not_exhausted" in manifest["failed_checks"]
    assert _read_jsonl(Path(manifest["actions_jsonl"])) == []


def test_route_exhaustion_plan_blocks_unsafe_route_expansion_summary_flags(tmp_path: Path) -> None:
    route_expansion = _route_expansion(tmp_path)
    payload = json.loads(route_expansion.read_text(encoding="utf-8"))
    payload["summary"]["thresholds_relaxed"] = True
    payload["summary"]["training_started"] = True
    route_expansion.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_full_pack_route_exhaustion_recovery_plan(
        machine_gate_json=_machine_gate(tmp_path),
        minus_audit_json=_minus_audit(tmp_path),
        minus_route_expansion_json=route_expansion,
        reference_gap_plan_json=_reference_gap(tmp_path),
        raw_reference_handoff_json=_recovery_handoff(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert manifest["plan_status"] == "blocked"
    assert "minus_route_expansion_summary_thresholds_relaxed_true" in manifest["failed_checks"]
    assert "minus_route_expansion_summary_training_started_true" in manifest["failed_checks"]
    assert _read_jsonl(Path(manifest["actions_jsonl"])) == []
