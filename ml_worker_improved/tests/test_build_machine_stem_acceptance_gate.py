from __future__ import annotations

import json
from pathlib import Path

from tools.build_machine_stem_acceptance_gate import build_machine_stem_acceptance_gate


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def source_master(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "src" / "master.json",
        {
            "manifest_type": "no_reference_vocal_minus_master_listen_preview",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "release_ready": False,
            "can_publish_as_result": False,
            "pair_decision": "pair_compatible",
            "pair_window_count": 5,
            "pair_compatibility_score": 0.91,
            "safety": {
                "training_started": False,
                "fine_tune_started": False,
                "models_changed": False,
                "model_registry_edited": False,
                "thresholds_relaxed": False,
                "cleanup_routes_added": False,
            },
        },
    )


FULL_STEMS = ("vocal", "minus", "drums", "bass", "other", "guitar", "piano")


def pack(
    tmp_path: Path,
    master: Path,
    *,
    stems: tuple[str, ...] = ("vocal", "minus"),
    unverified_stems: tuple[str, ...] = (),
    missing_audio_stems: tuple[str, ...] = (),
) -> Path:
    audio_root = tmp_path / "pack" / "audio" / "mst"
    for stem in stems:
        if stem not in missing_audio_stems:
            audio_path = audio_root / f"{stem}.flac"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(f"audio-{stem}".encode("ascii"))
    items = [
        {
            "schema": "aggregate_short_listen_pack_audio.v1",
            "audio_path": f"audio/mst/{stem}.flac",
            "source_manifest": str(master),
            "stem_type": stem,
            "quality_status": "listen_preview_only_not_semantic_verified" if stem in unverified_stems else "machine_quality_verified",
            "semantic_quality_verified": stem not in unverified_stems,
            "release_ready": False,
            "can_publish_as_result": False,
        }
        for stem in stems
    ]
    return write_json(
        tmp_path / "pack" / "pack.json",
        {
            "schema": "aggregate_short_listen_pack_manifest.v1",
            "delivery_status": "aggregate_validation_pending",
            "delivery_allowed": False,
            "presentation_status": "aggregate_validation_pending_not_for_listen_pack",
            "release_status": "aggregate_validation_pending",
            "aggregate_validation_ready": True,
            "listenable": False,
            "release_ready": False,
            "can_publish_as_result": False,
            "item_count": len(items),
            "items": items,
        },
    )


def validation(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "validation" / "validation.json",
        {
            "schema": "aggregate_short_listen_pack_validation.v1",
            "validation_status": "passed",
            "failed_checks": [],
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )


def minus_audit(tmp_path: Path, *, passed: bool) -> Path:
    rows_path = tmp_path / "audit" / "rows.jsonl"
    row = {
        "schema": "short_minus_quality_audit_row.v1",
        "short_audio_path": "audio/mst/minus.flac",
        "audit_decision": "passed" if passed else "blocked",
        "audit_blockers": [] if passed else ["minus_detail_gate_blocked"],
        "quality_warning_reasons": [] if passed else ["residual_vocal_watch"],
        "release_ready": False,
        "can_publish_as_result": False,
    }
    write_jsonl(rows_path, [row])
    return write_json(
        tmp_path / "audit" / "audit.json",
        {
            "schema": "short_minus_quality_audit_manifest.v1",
            "audit_status": "completed" if passed else "completed_with_blockers",
            "failed_checks": [] if passed else ["minus_audit_blocked:audio/mst/minus.flac:minus_detail_gate_blocked"],
            "minus_item_count": 1,
            "accepted_minus_count": 1 if passed else 0,
            "blocked_minus_count": 0 if passed else 1,
            "warning_count": 0 if passed else 1,
            "rows_jsonl": str(rows_path),
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )


def semantic_review(tmp_path: Path, *, pack_manifest: Path, stems: tuple[str, ...] = FULL_STEMS) -> Path:
    rows_path = tmp_path / "semantic" / "rows.jsonl"
    evidence_dir = tmp_path / "semantic" / "src"
    rows = []
    for stem in stems:
        evidence_path = evidence_dir / f"{stem}.json"
        write_json(
            evidence_path,
            {
                "manifest_type": "no_reference_stem_listen_preview",
                "source_manifest": str(pack_manifest),
                "source_stem_id": f"audio/mst/{stem}.flac",
                "stem_type": stem,
            },
        )
        rows.append(
            {
                "schema": "semantic_stem_quality_review_row.v1",
                "review_status": "accepted",
                "semantic_quality_status": "verified",
                "semantic_quality_verified": True,
                "stem_type": stem,
                "audio_path": f"/app/review/audio/t001/{stem}.flac",
                "evidence_paths": [str(evidence_path)],
                "blockers": [],
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    write_jsonl(rows_path, rows)
    return write_json(
        tmp_path / "semantic" / "review.json",
        {
            "schema": "semantic_stem_quality_review.v1",
            "review_status": "completed",
            "failed_checks": [],
            "rows_jsonl": str(rows_path),
            "semantic_quality_verified_item_count": len(rows),
            "release_ready": False,
            "can_publish_as_result": False,
        },
    )


def training_plan(tmp_path: Path, jobs: list[dict]) -> Path:
    return write_json(
        tmp_path / "training" / "plan.json",
        {
            "schema": "generic_audio_finetune_job_plan.v1",
            "manifest_type": "generic_audio_finetune_job_plan",
            "summary": {
                "job_count": len(jobs),
                "ready_job_count": sum(1 for job in jobs if job.get("status") == "ready_for_gpu_training"),
                "blocked_job_count": sum(1 for job in jobs if job.get("status") != "ready_for_gpu_training"),
            },
            "jobs": jobs,
            "delivery_allowed": False,
            "release_ready": False,
        },
    )


def train_ready_decision(tmp_path: Path, stem: str, *, train_ready: bool = True) -> Path:
    return write_json(
        tmp_path / "training" / f"{stem}_decision.json",
        {
            "schema": "train_ready_audio_model_decision_manifest.v1",
            "manifest_type": "train_ready_audio_model_decision",
            "decision": "train_result_verified" if train_ready else "train_result_not_ready",
            "train_ready": train_ready,
            "failed_checks": [] if train_ready else ["holdout_failed"],
            "best_matrix_row": {"stem": stem},
            "delivery_allowed": False,
            "release_ready": False,
            "promotion_allowed": False,
            "model_registry_edited": False,
            "runtime_gates_changed": False,
            "thresholds_relaxed": False,
        },
    )


def test_machine_stem_acceptance_gate_accepts_only_machine_passed_pack(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "accepted"
    assert manifest["final_result_allowed"] is True
    assert manifest["delivery_allowed"] is True
    assert manifest["failed_checks"] == []
    assert manifest["contract"]["no_human_listen_required"] is True
    assert manifest["contract"]["human_listen_pending_is_blocked"] is True


def test_machine_stem_acceptance_gate_blocks_incomplete_full_stem_pack(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=("vocal", "minus")),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["final_result_allowed"] is False
    assert manifest["missing_required_stems"] == ["bass", "drums", "guitar", "other", "piano"]
    assert "required_stem_missing:bass" in manifest["failed_checks"]
    assert "required_stem_missing:piano" in manifest["failed_checks"]


def test_machine_stem_acceptance_gate_blocks_missing_required_stem_audio_file(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS, missing_audio_stems=("drums",)),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["final_result_allowed"] is False
    assert "required_stem_audio_file_missing:drums" in manifest["failed_checks"]
    assert any(
        check["stem_type"] == "drums" and check["exists"] is False
        for check in manifest["stem_audio_file_checks"]
    )


def test_machine_stem_acceptance_gate_blocks_unverified_required_stems(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS, unverified_stems=("drums",)),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["final_result_allowed"] is False
    assert "required_stem_quality_not_verified:drums" in manifest["failed_checks"]


def test_machine_stem_acceptance_gate_uses_semantic_review_evidence(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    pack_manifest = pack(tmp_path, master, stems=FULL_STEMS, unverified_stems=("vocal", "drums", "bass", "other", "guitar", "piano"))
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack_manifest,
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        semantic_review_json=semantic_review(tmp_path, pack_manifest=pack_manifest),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "accepted"
    assert manifest["final_result_allowed"] is True
    assert manifest["unverified_required_stems"] == []
    assert manifest["semantic_review_status"] == "completed"


def test_machine_stem_acceptance_gate_blocks_minus_audit_failures(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=False),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["final_result_allowed"] is False
    assert manifest["delivery_allowed"] is False
    assert "minus_audit_not_completed" in manifest["failed_checks"]
    assert "minus_audit_failed_checks_present" in manifest["failed_checks"]
    assert "minus_audit_blocked_minus_present" in manifest["failed_checks"]
    assert "minus_audit_warnings_present" in manifest["failed_checks"]
    blockers = [json.loads(line) for line in Path(manifest["blockers_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert any(row["reason"] == "minus_audit_row_not_passed" for row in blockers)
    assert any(row["reason"] == "minus_audit_row_warnings_present" for row in blockers)


def test_machine_stem_acceptance_gate_blocks_listenable_staged_pack(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    pack_manifest = pack(tmp_path, master, stems=FULL_STEMS)
    payload = json.loads(pack_manifest.read_text(encoding="utf-8"))
    payload["listenable"] = True
    pack_manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack_manifest,
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["final_result_allowed"] is False
    assert "pack_listenable_not_false" in manifest["failed_checks"]


def test_machine_stem_acceptance_gate_blocks_missing_training_coverage_when_required(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
        require_training_coverage=True,
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["final_result_allowed"] is False
    assert manifest["require_training_coverage"] is True
    assert manifest["train_ready_stems"] == []
    assert "training_plan_missing" in manifest["failed_checks"]
    assert "required_stem_train_ready_missing:minus" in manifest["failed_checks"]
    assert "required_stem_train_ready_missing:piano" in manifest["failed_checks"]


def test_machine_stem_acceptance_gate_blocks_training_plan_job_blockers(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    plan = training_plan(
        tmp_path,
        [
            {
                "scope": "minus_teacher_mask_adapter",
                "status": "blocked_insufficient_unique_tracks",
                "stem_types": ["minus"],
                "unique_track_count": 1,
                "min_unique_tracks": 3,
            },
            {
                "scope": "stem_teacher_mask_adapter",
                "status": "blocked_insufficient_unique_tracks",
                "stem_types": ["drums", "bass", "other", "guitar", "piano"],
                "unique_track_count": 1,
                "min_unique_tracks": 3,
            },
            {
                "scope": "vocal_teacher_mask_adapter",
                "status": "ready_for_gpu_training",
                "stem_types": ["vocal"],
                "unique_track_count": 3,
                "min_unique_tracks": 3,
            },
            {
                "scope": "vocal_alternative_teacher_mask_adapter",
                "status": "blocked_insufficient_unique_tracks",
                "stem_types": ["vocal"],
                "unique_track_count": 1,
                "min_unique_tracks": 3,
            },
        ],
    )
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
        require_training_coverage=True,
        training_plan_json=plan,
        train_ready_decision_jsons=[train_ready_decision(tmp_path, "vocal")],
    )

    assert manifest["gate_status"] == "blocked"
    assert manifest["train_ready_stems"] == ["vocal"]
    assert manifest["training_ready_job_stems"] == ["vocal"]
    assert "training_job_blocked:minus:blocked_insufficient_unique_tracks" in manifest["failed_checks"]
    assert "training_job_blocked:piano:blocked_insufficient_unique_tracks" in manifest["failed_checks"]
    assert "training_job_blocked:vocal:blocked_insufficient_unique_tracks" not in manifest["failed_checks"]
    assert "required_stem_train_ready_missing:minus" in manifest["failed_checks"]
    assert "required_stem_train_ready_missing:piano" in manifest["failed_checks"]


def test_machine_stem_acceptance_gate_accepts_when_required_training_coverage_is_verified(tmp_path: Path) -> None:
    master = source_master(tmp_path)
    plan = training_plan(
        tmp_path,
        [
            {"scope": f"{stem}_teacher_mask_adapter", "status": "ready_for_gpu_training", "stem_types": [stem]}
            for stem in FULL_STEMS
        ],
    )
    manifest = build_machine_stem_acceptance_gate(
        pack_manifest_json=pack(tmp_path, master, stems=FULL_STEMS),
        validation_json=validation(tmp_path),
        minus_audit_json=minus_audit(tmp_path, passed=True),
        output_dir=tmp_path / "gate",
        require_training_coverage=True,
        training_plan_json=plan,
        train_ready_decision_jsons=[train_ready_decision(tmp_path, stem) for stem in FULL_STEMS],
    )

    assert manifest["gate_status"] == "accepted"
    assert manifest["final_result_allowed"] is True
    assert manifest["train_ready_stems"] == sorted(FULL_STEMS)
    assert manifest["failed_checks"] == []
