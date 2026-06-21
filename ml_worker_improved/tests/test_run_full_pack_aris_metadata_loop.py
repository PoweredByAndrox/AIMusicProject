from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_full_pack_aris_metadata_loop import run_full_pack_aris_metadata_loop


FULL_STEMS = ["vocal", "minus", "drums", "bass", "other", "guitar", "piano"]
REQUIRED_GATES = [
    "aggregate_pack_validation_passed",
    "full_required_stem_coverage",
    "required_stem_machine_quality_evidence",
    "source_vocal_minus_master_pair_compatible",
    "minus_quality_audit_passed_without_warnings",
    "required_stem_train_ready_model_evidence",
]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return str(path)


def _training_row(tmp_path: Path, *, track_id: str = "track-a", stem: str = "vocal") -> dict:
    return {
        "schema": "generic_finetuning_audio_training_example.v1",
        "id": f"{track_id}-{stem}",
        "track_id": track_id,
        "stem_type": stem,
        "source_audio_path": _audio(tmp_path / "audio" / track_id / "mix.flac"),
        "target_audio_path": _audio(tmp_path / "audio" / track_id / f"{stem}.flac"),
        "target_audio_sha256": f"sha-{track_id}-{stem}",
        "permission_status": "project_owned",
        "trainable": True,
        "reference_verified": True,
        "release_ready": False,
        "delivery_allowed": False,
    }


def _base_examples(tmp_path: Path) -> Path:
    return _write_jsonl(tmp_path / "base.jsonl", [_training_row(tmp_path)])


def _gap_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": "minus",
                "required_reference_stem_count": 2,
                "missing_unique_track_count": 2,
                "missing_training_job": False,
                "priority": "high",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g002",
                "queue_item_id": "q002",
                "stem_type": "drums",
                "required_reference_stem_count": 2,
                "missing_unique_track_count": 2,
                "missing_training_job": True,
                "priority": "normal",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 2,
            "missing_reference_stem_count": 4,
            "missing_training_job_count": 1,
            "failed_checks": [],
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _machine_gate(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "gate" / "gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "gate_status": "blocked",
            "final_result_allowed": False,
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "failed_checks": ["required_stem_train_ready_missing:minus"],
            "required_stems": FULL_STEMS,
            "stem_types": FULL_STEMS,
            "missing_required_stems": [],
            "required_machine_gates": REQUIRED_GATES,
            "require_training_coverage": True,
            "train_ready_stems": ["vocal"],
            "contract": {
                "training_coverage_required": True,
                "train_ready_model_evidence_required": True,
            },
            "release_ready": False,
            "training_started": False,
        },
    )


def _repair_queue(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "repair" / "queue.json",
        {
            "schema": "short_listen_repair_queue_manifest.v1",
            "queue_status": "ready_for_full_pack_loop",
            "failed_checks": [],
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _loop_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "loop" / "rows.jsonl",
        [
            {
                "schema": "full_pack_loop_iteration_action.v1",
                "action_id": "a001",
                "stem_type": "minus",
                "action_type": "collect_verified_reference_stem",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "loop" / "plan.json",
        {
            "schema": "full_pack_loop_iteration_plan.v1",
            "plan_status": "blocked_waiting_for_verified_reference_corpus",
            "failed_checks": [],
            "rows_jsonl": str(rows),
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_without_permission(tmp_path: Path, *, track_id: str, stem: str) -> Path:
    root = tmp_path / "candidates" / track_id
    _audio(root / "source_mix.flac")
    _audio(root / "stems" / f"{stem}.flac")
    _write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": track_id,
            "ground_truth_contract": {
                "verified": True,
                "source": "FL Studio split mixer tracks plus full mix supplied by user",
            },
            "mix": {"path": "source_mix.flac"},
            "stems": {stem: {"path": f"stems/{stem}.flac"}},
        },
    )
    return tmp_path / "candidates"


def _bass_gap_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": "bass",
                "required_reference_stem_count": 1,
                "missing_unique_track_count": 1,
                "missing_training_job": True,
                "priority": "high",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 1,
            "missing_reference_stem_count": 1,
            "missing_training_job_count": 1,
            "failed_checks": [],
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _bass_guitar_gap_plan(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "gap" / "rows.jsonl",
        [
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": "bass",
                "required_reference_stem_count": 1,
                "missing_unique_track_count": 1,
                "missing_training_job": True,
                "priority": "high",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
            {
                "schema": "full_pack_reference_corpus_gap.v1",
                "gap_id": "g002",
                "queue_item_id": "q002",
                "stem_type": "guitar",
                "required_reference_stem_count": 1,
                "missing_unique_track_count": 1,
                "missing_training_job": True,
                "priority": "normal",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "gap" / "gap.json",
        {
            "schema": "full_pack_reference_corpus_gap_plan.v1",
            "gap_status": "blocked_waiting_for_verified_reference_corpus",
            "gap_count": 2,
            "missing_reference_stem_count": 2,
            "missing_training_job_count": 2,
            "failed_checks": [],
            "rows_jsonl": str(rows),
            "outputs": {"rows_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _raw_flstudio_bass_export(tmp_path: Path) -> Path:
    root = tmp_path / "golden_audio" / "track-b"
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    frame_count = sample_rate // 2
    time = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    bass = 0.18 * np.sin(2 * np.pi * 90 * time)
    sf.write(root / "00_source_mix.flac", np.column_stack([bass, bass]), sample_rate, subtype="PCM_24")
    sf.write(root / "01_bass_reference_teacher.flac", np.column_stack([bass, bass]), sample_rate, subtype="PCM_24")
    return tmp_path / "golden_audio"


def _verified_clean_bundle(root: Path, *, stem: str) -> Path:
    _audio(root / "source_mix.flac")
    _audio(root / "stems" / f"{stem}.flac")
    _write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": f"clean-{stem}",
            "ground_truth_contract": {
                "verified": True,
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


def _bass_guitar_handoff(tmp_path: Path) -> Path:
    rows = _write_jsonl(
        tmp_path / "hand" / "items.jsonl",
        [
            {
                "schema": "full_pack_reference_bundle_handoff_item.v1",
                "bundle_alias": "r001",
                "source_collection_id": "c001",
                "gap_id": "g001",
                "queue_item_id": "q001",
                "stem_type": "bass",
                "slot_index": 1,
                "missing_training_job": True,
                "required_directory": "refs/r001",
                "required_files": ["source_mix.flac", "stems/bass.flac", "ground_truth_manifest.json"],
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
            {
                "schema": "full_pack_reference_bundle_handoff_item.v1",
                "bundle_alias": "r002",
                "source_collection_id": "c002",
                "gap_id": "g002",
                "queue_item_id": "q002",
                "stem_type": "guitar",
                "slot_index": 1,
                "missing_training_job": True,
                "required_directory": "refs/r002",
                "required_files": ["source_mix.flac", "stems/guitar.flac", "ground_truth_manifest.json"],
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            },
        ],
    )
    return _write_json(
        tmp_path / "hand" / "handoff.json",
        {
            "schema": "full_pack_reference_bundle_handoff_manifest.v1",
            "handoff_status": "ready_for_reference_bundle_collection",
            "handoff_item_count": 2,
            "failed_checks": [],
            "outputs": {"items_jsonl": str(rows)},
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def _blocked_recovery_material_handoff(tmp_path: Path) -> Path:
    items = _write_jsonl(
        tmp_path / "rhand" / "items.jsonl",
        [
            {
                "schema": "full_pack_raw_reference_recovery_material_handoff_item.v1",
                "action_id": "a001",
                "candidate_root": "",
                "delivery_allowed": False,
                "release_ready": False,
                "training_started": False,
            }
        ],
    )
    return _write_json(
        tmp_path / "rhand" / "handoff.json",
        {
            "schema": "full_pack_raw_reference_recovery_material_handoff_manifest.v1",
            "manifest_type": "full_pack_raw_reference_recovery_material_handoff",
            "handoff_status": "blocked_waiting_for_recovery_reference_material",
            "failed_checks": [],
            "handoff_item_count": 1,
            "ready_handoff_count": 0,
            "blocked_handoff_count": 1,
            "candidate_roots": [],
            "outputs": {
                "items_jsonl": str(items),
                "manifest_json": str(tmp_path / "rhand" / "handoff.json"),
            },
            "delivery_allowed": False,
            "release_ready": False,
            "training_started": False,
        },
    )


def test_metadata_loop_refreshes_reference_chain_and_stays_blocked_without_new_refs(tmp_path: Path) -> None:
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[tmp_path / "empty_candidates"],
        output_dir=tmp_path / "out",
        min_unique_tracks=3,
    )

    assert manifest["schema"] == "full_pack_aris_metadata_loop_manifest.v1"
    assert manifest["loop_status"] == "blocked_continuing_aris_loop"
    assert manifest["goal_complete"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["training_started"] is False
    assert manifest["accepted_reference_rows"] == 0
    assert manifest["remaining_reference_stem_count"] == 4
    assert manifest["cycle_status"] == "blocked_waiting_for_verified_reference_corpus"
    assert manifest["next_action"] == "collect_verified_reference_corpus"

    outputs = manifest["outputs"]
    for key in (
        "intake_json",
        "merge_json",
        "validation_json",
        "training_plan_json",
        "cycle_status_json",
        "collection_queue_json",
        "reference_bundle_handoff_json",
        "reference_bundle_scaffold_json",
        "reference_bundle_preflight_json",
        "reference_material_discovery_json",
        "raw_reference_candidates_json",
        "raw_reference_worklist_json",
        "next_run_plan_json",
        "completion_audit_json",
    ):
        assert Path(outputs[key]).exists(), key
    assert Path(outputs["manifest_json"]).name == "loop.json"
    assert all(len(path.name) <= 32 for path in (tmp_path / "out").iterdir() if path.is_file())

    audit = _read_json(Path(outputs["completion_audit_json"]))
    assert audit["goal_complete"] is False
    assert "verified_corpus_not_passed" in audit["remaining_blockers"]

    collection = _read_json(Path(outputs["collection_queue_json"]))
    assert collection["queue_status"] == "ready_for_reference_collection"
    assert collection["remaining_by_stem"] == {"drums": 2, "minus": 2}

    handoff = _read_json(Path(outputs["reference_bundle_handoff_json"]))
    assert handoff["handoff_status"] == "ready_for_reference_bundle_collection"
    assert handoff["handoff_item_count"] == 4
    scaffold = _read_json(Path(outputs["reference_bundle_scaffold_json"]))
    assert scaffold["scaffold_status"] == "created_reference_bundle_scaffold"
    assert scaffold["bundle_directory_count"] == 4
    assert manifest["reference_bundle_scaffold_status"] == "created_reference_bundle_scaffold"
    preflight = _read_json(Path(outputs["reference_bundle_preflight_json"]))
    assert preflight["preflight_status"] == "blocked_waiting_for_reference_bundles"
    assert preflight["ready_bundle_count"] == 0
    assert preflight["blocked_bundle_count"] == 4
    assert preflight["candidate_roots"] == []
    assert manifest["reference_bundle_handoff_status"] == "ready_for_reference_bundle_collection"
    assert manifest["reference_bundle_preflight_status"] == "blocked_waiting_for_reference_bundles"
    discovery = _read_json(Path(outputs["reference_material_discovery_json"]))
    assert discovery["discovery_status"] == "blocked_no_eligible_reference_material"
    assert discovery["eligible_candidate_rows"] == 0
    assert manifest["reference_material_discovery_status"] == "blocked_no_eligible_reference_material"
    assert manifest["eligible_reference_material_rows"] == 0
    raw_candidates = _read_json(Path(outputs["raw_reference_candidates_json"]))
    assert raw_candidates["raw_candidate_status"] == "blocked_no_raw_reference_candidates"
    assert raw_candidates["accepted_reference_rows"] == 0
    assert manifest["raw_reference_candidate_status"] == "blocked_no_raw_reference_candidates"
    assert manifest["raw_reference_candidate_rows"] == 0
    raw_worklist = _read_json(Path(outputs["raw_reference_worklist_json"]))
    assert raw_worklist["worklist_status"] == "no_raw_reference_bundle_work_items"
    assert raw_worklist["work_item_count"] == 0
    assert manifest["raw_reference_worklist_status"] == "no_raw_reference_bundle_work_items"
    assert manifest["raw_reference_work_item_count"] == 0
    next_run = _read_json(Path(outputs["next_run_plan_json"]))
    assert next_run["next_run_status"] == "blocked_waiting_for_reference_bundles"
    assert next_run["candidate_roots"] == []
    assert next_run["next_command_args"] == []
    assert manifest["next_run_status"] == "blocked_waiting_for_reference_bundles"
    assert manifest["next_run_candidate_root_count"] == 0


def test_metadata_loop_can_import_verified_raw_reference_bundles(tmp_path: Path) -> None:
    raw_root = _raw_flstudio_bass_export(tmp_path)
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_bass_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[raw_root],
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
        raw_reference_allowed_source_roots=[raw_root],
        raw_reference_rights_status="project_owned",
        raw_reference_require_full_mix_reconstruction=True,
    )

    assert manifest["loop_status"] == "blocked_continuing_aris_loop"
    assert manifest["raw_reference_verified_bundle_status"] == "verified_reference_bundles_ready"
    assert manifest["raw_reference_verified_bundle_count"] == 1
    assert manifest["raw_reference_verified_blocked_bundle_count"] == 0
    assert manifest["accepted_reference_rows"] == 1
    assert manifest["remaining_reference_stem_count"] == 0
    assert manifest["safety"]["audio_read"] is True
    assert manifest["safety"]["audio_copied"] is True

    raw_verified = _read_json(Path(manifest["outputs"]["raw_reference_verified_bundles_json"]))
    assert raw_verified["candidate_roots"]
    assert Path(raw_verified["candidate_roots"][0]).name == "b001"
    combined = _read_json(Path(manifest["outputs"]["combined_intake_json"]))
    assert combined["accepted_by_stem"]["bass"] == 1


def test_metadata_loop_uses_handoff_slots_for_raw_reference_worklist(tmp_path: Path) -> None:
    raw_root = _raw_flstudio_bass_export(tmp_path)
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_bass_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[raw_root],
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
        raw_reference_handoff_json=_bass_guitar_handoff(tmp_path),
        raw_reference_allowed_source_roots=[raw_root],
        raw_reference_rights_status="project_owned",
        raw_reference_require_full_mix_reconstruction=True,
    )

    raw_worklist = _read_json(Path(manifest["outputs"]["raw_reference_worklist_json"]))
    assert raw_worklist["slot_aware"] is True
    assert raw_worklist["formal_slot_count"] == 2
    assert raw_worklist["matched_raw_candidate_count"] == 1
    assert raw_worklist["missing_raw_candidate_slot_count"] == 1
    assert raw_worklist["work_items_by_status"] == {
        "blocked_missing_raw_reference_candidate": 1,
        "needs_verified_ground_truth_manifest": 1,
    }

    raw_verified = _read_json(Path(manifest["outputs"]["raw_reference_verified_bundles_json"]))
    assert raw_verified["verified_bundle_count"] == 1
    assert raw_verified["blocked_bundle_count"] == 1
    assert raw_verified["blocked_by_stem"] == {"guitar": 1}
    assert raw_verified["ready_by_stem"] == {"bass": 1}
    blocked = _read_jsonl(Path(raw_verified["outputs"]["blocked_jsonl"]))
    assert blocked[0]["bundle_alias"] == "r002"
    assert blocked[0]["blockers"] == ["missing_raw_reference_candidate"]
    assert blocked[0]["ground_truth_manifest_json"] == ""
    raw_recovery = _read_json(Path(manifest["outputs"]["raw_reference_recovery_queue_json"]))
    assert raw_recovery["queue_status"] == "ready_for_raw_reference_recovery"
    assert raw_recovery["queue_item_count"] == 1
    assert raw_recovery["route_intent_counts"] == {"replace_missing_requested_stem_candidate": 1}
    assert raw_recovery["delivery_allowed"] is False
    assert manifest["raw_reference_recovery_queue_status"] == "ready_for_raw_reference_recovery"
    assert manifest["raw_reference_recovery_queue_item_count"] == 1
    assert manifest["raw_reference_recovery_route_intent_counts"] == {"replace_missing_requested_stem_candidate": 1}
    raw_recovery_action_plan = _read_json(Path(manifest["outputs"]["raw_reference_recovery_action_plan_json"]))
    assert raw_recovery_action_plan["action_plan_status"] == "ready_for_raw_reference_recovery_actions"
    assert raw_recovery_action_plan["action_item_count"] == 1
    assert raw_recovery_action_plan["route_intent_counts"] == {"replace_missing_requested_stem_candidate": 1}
    assert raw_recovery_action_plan["delivery_allowed"] is False
    assert manifest["raw_reference_recovery_action_plan_status"] == "ready_for_raw_reference_recovery_actions"
    assert manifest["raw_reference_recovery_action_item_count"] == 1
    assert manifest["raw_reference_recovery_action_route_counts"] == {"replace_missing_requested_stem_candidate": 1}
    raw_recovery_execution = _read_json(Path(manifest["outputs"]["raw_reference_recovery_execution_json"]))
    assert raw_recovery_execution["execution_status"] == "blocked_waiting_for_external_reference_material"
    assert raw_recovery_execution["action_item_count"] == 1
    assert raw_recovery_execution["execution_status_counts"] == {
        "blocked_waiting_for_missing_requested_stem_candidate": 1
    }
    assert raw_recovery_execution["delivery_allowed"] is False
    assert manifest["raw_reference_recovery_execution_status"] == "blocked_waiting_for_external_reference_material"
    assert manifest["raw_reference_recovery_execution_item_count"] == 1
    raw_recovery_material_responses = _read_json(
        Path(manifest["outputs"]["raw_reference_recovery_material_responses_json"])
    )
    assert raw_recovery_material_responses["response_status"] == "blocked_waiting_for_recovery_reference_material"
    assert raw_recovery_material_responses["action_item_count"] == 1
    assert raw_recovery_material_responses["ready_response_count"] == 0
    assert raw_recovery_material_responses["blocked_response_count"] == 1
    assert raw_recovery_material_responses["written_response_count"] == 0
    assert raw_recovery_material_responses["delivery_allowed"] is False
    assert manifest["raw_reference_recovery_material_response_status"] == (
        "blocked_waiting_for_recovery_reference_material"
    )
    assert manifest["raw_reference_recovery_material_response_ready_count"] == 0
    assert manifest["raw_reference_recovery_material_response_blocked_count"] == 1
    raw_recovery_material_preflight = _read_json(
        Path(manifest["outputs"]["raw_reference_recovery_material_preflight_json"])
    )
    assert raw_recovery_material_preflight["preflight_status"] == "blocked_waiting_for_recovery_reference_material"
    assert raw_recovery_material_preflight["action_item_count"] == 1
    assert raw_recovery_material_preflight["ready_action_count"] == 0
    assert raw_recovery_material_preflight["blocked_action_count"] == 1
    assert raw_recovery_material_preflight["candidate_roots"] == []
    assert raw_recovery_material_preflight["delivery_allowed"] is False
    assert manifest["raw_reference_recovery_material_preflight_status"] == (
        "blocked_waiting_for_recovery_reference_material"
    )
    assert manifest["raw_reference_recovery_material_ready_count"] == 0
    assert manifest["raw_reference_recovery_material_blocked_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["result_delivery_allowed"] is False


def test_metadata_loop_skips_incompatible_raw_reference_handoff_without_schema_blocking(tmp_path: Path) -> None:
    raw_root = _raw_flstudio_bass_export(tmp_path)
    recovery_handoff = _blocked_recovery_material_handoff(tmp_path)

    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_bass_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[raw_root],
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
        raw_reference_handoff_json=recovery_handoff,
        raw_reference_allowed_source_roots=[raw_root],
        raw_reference_rights_status="project_owned",
        raw_reference_require_full_mix_reconstruction=True,
    )

    raw_worklist = _read_json(Path(manifest["outputs"]["raw_reference_worklist_json"]))
    assert raw_worklist["worklist_status"] == "no_raw_reference_bundle_work_items"
    assert raw_worklist["failed_checks"] == []
    assert raw_worklist["work_item_count"] == 0
    assert raw_worklist["raw_reference_handoff_use_status"] == "skipped_incompatible_handoff"
    assert raw_worklist["raw_reference_handoff_skip_reason"] == "raw_reference_handoff_not_ready_for_bundle_worklist"
    assert raw_worklist["handoff_json"] == str(recovery_handoff)
    assert "handoff_schema_unexpected" not in raw_worklist["failed_checks"]
    assert "handoff_status_not_ready" not in raw_worklist["failed_checks"]

    raw_verified = _read_json(Path(manifest["outputs"]["raw_reference_verified_bundles_json"]))
    assert raw_verified["bundle_status"] == "no_raw_reference_bundle_items"
    assert raw_verified["failed_checks"] == []
    assert raw_verified["verified_bundle_count"] == 0
    assert raw_verified["blocked_bundle_count"] == 0
    assert raw_verified["safety"]["audio_read"] is False
    assert raw_verified["safety"]["audio_copied"] is False

    assert manifest["raw_reference_handoff_use_status"] == "skipped_incompatible_handoff"
    assert manifest["raw_reference_handoff_skip_reason"] == "raw_reference_handoff_not_ready_for_bundle_worklist"
    assert manifest["raw_reference_worklist_status"] == "no_raw_reference_bundle_work_items"
    assert manifest["raw_reference_work_item_count"] == 0
    assert manifest["delivery_allowed"] is False
    assert manifest["result_delivery_allowed"] is False


def test_metadata_loop_builds_raw_recovery_responses_from_clean_material_roots(tmp_path: Path) -> None:
    raw_root = _raw_flstudio_bass_export(tmp_path)
    recovery_material_root = _verified_clean_bundle(tmp_path / "clean" / "g001", stem="guitar")
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_bass_guitar_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[raw_root],
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
        raw_reference_handoff_json=_bass_guitar_handoff(tmp_path),
        raw_reference_allowed_source_roots=[raw_root],
        raw_reference_recovery_material_candidate_roots=[recovery_material_root],
        raw_reference_rights_status="project_owned",
        raw_reference_require_full_mix_reconstruction=True,
    )

    responses = _read_json(Path(manifest["outputs"]["raw_reference_recovery_material_responses_json"]))
    assert responses["response_status"] == "ready_for_raw_reference_recovery_material_preflight"
    assert responses["ready_response_count"] == 1
    assert responses["blocked_response_count"] == 0
    assert responses["response_paths"] == [str(tmp_path / "out" / "rexec" / "rsp" / "a001.json")]
    preflight = _read_json(Path(manifest["outputs"]["raw_reference_recovery_material_preflight_json"]))
    assert preflight["preflight_status"] == "ready_for_raw_reference_recovery_verification"
    assert preflight["ready_action_count"] == 1
    assert preflight["blocked_action_count"] == 0
    assert preflight["candidate_roots"] == [str(recovery_material_root)]
    handoff = _read_json(Path(manifest["outputs"]["raw_reference_recovery_material_handoff_json"]))
    assert handoff["handoff_status"] == "ready_for_reference_corpus_intake"
    assert handoff["candidate_roots"] == [str(recovery_material_root)]
    assert handoff["delivery_allowed"] is False
    combined = _read_json(Path(manifest["outputs"]["combined_intake_json"]))
    assert combined["accepted_by_stem"] == {"bass": 1, "guitar": 1}
    assert combined["remaining_reference_stem_count"] == 0
    assert manifest["raw_reference_recovery_material_response_status"] == (
        "ready_for_raw_reference_recovery_material_preflight"
    )
    assert manifest["raw_reference_recovery_material_preflight_status"] == (
        "ready_for_raw_reference_recovery_verification"
    )
    assert manifest["raw_reference_recovery_material_handoff_status"] == "ready_for_reference_corpus_intake"
    assert manifest["raw_reference_recovery_material_handoff_ready_count"] == 1
    assert manifest["accepted_reference_rows"] == 2
    assert manifest["remaining_reference_stem_count"] == 0
    next_run = _read_json(Path(manifest["outputs"]["next_run_plan_json"]))
    assert next_run["raw_reference_recovery_material_candidate_roots"] == [str(recovery_material_root)]
    assert str(recovery_material_root) in next_run["candidate_roots"]
    assert "--candidate-root" in next_run["next_command_args"]
    assert str(recovery_material_root) in next_run["next_command_args"]
    assert manifest["delivery_allowed"] is False
    assert manifest["result_delivery_allowed"] is False


def test_metadata_loop_uses_allowed_roots_as_recovery_material_candidates(tmp_path: Path) -> None:
    raw_root = _raw_flstudio_bass_export(tmp_path)
    recovery_material_root = _verified_clean_bundle(tmp_path / "clean" / "g001", stem="guitar")

    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_bass_guitar_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[raw_root],
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
        raw_reference_handoff_json=_bass_guitar_handoff(tmp_path),
        raw_reference_allowed_source_roots=[raw_root, recovery_material_root],
        raw_reference_rights_status="project_owned",
        raw_reference_require_full_mix_reconstruction=True,
    )

    responses = _read_json(Path(manifest["outputs"]["raw_reference_recovery_material_responses_json"]))
    assert responses["candidate_roots"] == [str(raw_root), str(recovery_material_root)]
    assert responses["response_status"] == "ready_for_raw_reference_recovery_material_preflight"
    assert responses["ready_response_count"] == 1
    assert responses["blocked_response_count"] == 0
    assert responses["response_paths"] == [str(tmp_path / "out" / "rexec" / "rsp" / "a001.json")]
    preflight = _read_json(Path(manifest["outputs"]["raw_reference_recovery_material_preflight_json"]))
    assert preflight["preflight_status"] == "ready_for_raw_reference_recovery_verification"
    assert preflight["candidate_roots"] == [str(recovery_material_root)]
    handoff = _read_json(Path(manifest["outputs"]["raw_reference_recovery_material_handoff_json"]))
    assert handoff["handoff_status"] == "ready_for_reference_corpus_intake"
    assert handoff["candidate_roots"] == [str(recovery_material_root)]
    assert manifest["raw_reference_recovery_material_response_status"] == (
        "ready_for_raw_reference_recovery_material_preflight"
    )
    assert manifest["raw_reference_recovery_material_handoff_status"] == "ready_for_reference_corpus_intake"
    assert manifest["delivery_allowed"] is False
    assert manifest["result_delivery_allowed"] is False


def test_metadata_loop_carries_raw_reference_evidence_into_next_collection_handoff(tmp_path: Path) -> None:
    raw_root = _raw_flstudio_bass_export(tmp_path)
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_bass_guitar_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[raw_root],
        output_dir=tmp_path / "out",
        min_unique_tracks=1,
        raw_reference_handoff_json=_bass_guitar_handoff(tmp_path),
        raw_reference_allowed_source_roots=[raw_root],
        raw_reference_rights_status="project_owned",
        raw_reference_require_full_mix_reconstruction=True,
    )

    assert manifest["loop_status"] == "blocked_continuing_aris_loop"
    assert manifest["remaining_reference_stem_count"] == 1
    assert manifest["delivery_allowed"] is False
    assert manifest["result_delivery_allowed"] is False

    collection = _read_json(Path(manifest["outputs"]["collection_queue_json"]))
    assert collection["queue_status"] == "ready_for_reference_collection"
    assert collection["remaining_by_stem"] == {"guitar": 1}
    assert collection["raw_reference_evidence_item_count"] == 1
    collection_rows = _read_jsonl(Path(collection["outputs"]["items_jsonl"]))
    assert len(collection_rows) == 1
    collection_evidence = collection_rows[0]["raw_reference_evidence"]
    assert collection_evidence["work_item_status"] == "blocked_missing_raw_reference_candidate"
    assert collection_evidence["blockers"] == ["missing_raw_reference_candidate"]
    assert collection_evidence["accepted_as_ground_truth"] is False

    handoff = _read_json(Path(manifest["outputs"]["reference_bundle_handoff_json"]))
    handoff_rows = _read_jsonl(Path(handoff["outputs"]["items_jsonl"]))
    assert len(handoff_rows) == 1
    handoff_evidence = handoff_rows[0]["raw_reference_evidence"]
    assert handoff_evidence["work_item_status"] == "blocked_missing_raw_reference_candidate"
    assert handoff_evidence["requires_clean_reference_bundle"] is True
    assert handoff_evidence["accepted_as_ground_truth"] is False


def test_metadata_loop_imports_permission_recovered_reference_rows(tmp_path: Path) -> None:
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=_base_examples(tmp_path),
        candidate_roots=[_candidate_without_permission(tmp_path, track_id="track-b", stem="minus")],
        output_dir=tmp_path / "out",
        min_unique_tracks=3,
    )

    assert manifest["loop_status"] == "blocked_continuing_aris_loop"
    assert manifest["goal_complete"] is False
    assert manifest["accepted_reference_rows"] == 1
    assert manifest["recovered_audio_training_examples"] == 1
    assert manifest["remaining_reference_stem_count"] == 3

    outputs = manifest["outputs"]
    assert Path(outputs["recovery_queue_jsons"][0]).name == "rec.json"
    assert Path(outputs["permission_decision_jsons"][0]).name == "perm.json"
    assert Path(outputs["recovery_ingest_jsons"][0]).name == "ingest.json"
    assert Path(outputs["combined_intake_json"]).name == "intake.json"

    permission = _read_json(Path(outputs["permission_decision_jsons"][0]))
    assert permission["decision_rows"] == 1
    ingest = _read_json(Path(outputs["recovery_ingest_jsons"][0]))
    assert ingest["summary"]["approved_audio_training_examples"] == 1

    combined = _read_json(Path(outputs["combined_intake_json"]))
    assert combined["accepted_by_stem"]["minus"] == 1
    assert combined["remaining_by_stem"] == {"drums": 2, "minus": 1}

    merge = _read_json(Path(outputs["merge_json"]))
    merged_rows = _read_jsonl(Path(merge["outputs"]["merged_audio_training_examples_jsonl"]))
    assert any(row["track_id"] == "track-b" and row["stem_type"] == "minus" for row in merged_rows)

    collection = _read_json(Path(outputs["collection_queue_json"]))
    assert collection["remaining_by_stem"] == {"drums": 2, "minus": 1}


def test_metadata_loop_does_not_count_duplicate_recovered_track_stems(tmp_path: Path) -> None:
    existing = _write_jsonl(tmp_path / "base.jsonl", [_training_row(tmp_path, track_id="track-a", stem="minus")])
    manifest = run_full_pack_aris_metadata_loop(
        machine_gate_json=_machine_gate(tmp_path),
        repair_queue_json=_repair_queue(tmp_path),
        full_pack_loop_plan_json=_loop_plan(tmp_path),
        reference_gap_plan_json=_gap_plan(tmp_path),
        existing_audio_training_examples_jsonl=existing,
        candidate_roots=[_candidate_without_permission(tmp_path, track_id="track-a", stem="minus")],
        output_dir=tmp_path / "out",
        min_unique_tracks=3,
    )

    assert manifest["loop_status"] == "blocked_continuing_aris_loop"
    assert manifest["recovery_ingest_audio_training_examples"] == 1
    assert manifest["recovered_audio_training_examples"] == 0
    assert manifest["remaining_reference_stem_count"] == 4

    combined = _read_json(Path(manifest["outputs"]["combined_intake_json"]))
    assert combined["excluded_recovered_audio_training_examples"] == 1
    assert combined["excluded_recovered_reason_counts"] == {"duplicate_existing_track_stem": 1}
    assert combined["remaining_by_stem"] == {"drums": 2, "minus": 2}
    assert combined["safety"]["metadata_only"] is False
    assert combined["safety"]["audio_read"] is True
    assert combined["safety"]["audio_sha256_read_by_recovery_ingest"] is True

    merge = _read_json(Path(manifest["outputs"]["merge_json"]))
    merged_rows = _read_jsonl(Path(merge["outputs"]["merged_audio_training_examples_jsonl"]))
    assert sum(1 for row in merged_rows if row["track_id"] == "track-a" and row["stem_type"] == "minus") == 1
