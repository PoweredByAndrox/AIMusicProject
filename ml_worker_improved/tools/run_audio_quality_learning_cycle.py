from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    LEARNING_CYCLE_SCHEMA,
    PROGRESS_FILE,
    REVIEW_FILE,
    SAFETY_FLAGS,
    append_progress,
    iter_jsonl,
    read_json,
    write_json,
    write_jsonl,
)
from tools.build_artifact_rag_metadata_overlay import build_artifact_rag_metadata_overlay
from tools.build_audio_quality_critic_dataset import build_audio_quality_critic_dataset
from tools.build_audio_quality_source_lock import build_audio_quality_source_lock
from tools.build_audio_quality_verified_corpus_recovery_plan import build_audio_quality_verified_corpus_recovery_plan
from tools.build_audio_quality_verified_material_request_packet import build_audio_quality_verified_material_request_packet
from tools.build_audio_quality_verified_rows import build_audio_quality_verified_rows
from tools.discover_audio_quality_promotion_candidates import discover_audio_quality_promotion_candidates
from tools.discover_audio_quality_verified_corpora import discover_audio_quality_verified_corpora
from tools.docker_runtime import require_docker_runtime
from tools.ingest_audio_quality_review_decisions import ingest_audio_quality_review_decisions


OUTCOME_FILE_NAMES = ("audio_quality_learning_outcome.jsonl", "learning_outcome.jsonl")
VERIFIED_OUTCOME_LABEL_TYPES = {"reference_verified", "human_verified", "locked_holdout"}
LEARNING_HANDOFF_SCHEMA = "audio_quality_learning_handoff.v1"
VERIFIED_ROW_INTAKE_SCHEMA = "audio_quality_verified_row_intake_batch.v1"


def _outcome_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in OUTCOME_FILE_NAMES:
            yield path


def _collect_outcomes(outcomes_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    verified: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    raw_excluded = 0
    for path in _outcome_files(outcomes_root):
        for row in iter_jsonl(path):
            source_row = {**row, "source_learning_outcome_jsonl": str(path)}
            if row.get("raw_prediction_only") is True:
                raw_excluded += 1
                review.append({**source_row, "review_reason": "raw_prediction_only", "delivery_allowed": False, "release_ready": False})
                continue
            label_type = str(row.get("label_type") or "")
            if row.get("verified_outcome") is True and label_type in VERIFIED_OUTCOME_LABEL_TYPES:
                verified.append(source_row)
            else:
                review.append({**source_row, "review_reason": "not_verified_for_training", "delivery_allowed": False, "release_ready": False})
    return verified, review, raw_excluded


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _metadata_overlay_include_roots(
    *,
    rag_current_json: Path,
    output_dir: Path,
    metadata_roots: Sequence[Path],
    metadata_overlay_output_dir: Path,
) -> list[Path]:
    artifact_rag_root = rag_current_json.parent
    overlay_resolved = metadata_overlay_output_dir.resolve()
    include_roots: list[Path] = []
    seen: set[Path] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            include_roots.append(path)
            seen.add(resolved)

    for metadata_root in metadata_roots:
        add_root(metadata_root)
    if _is_inside(output_dir, artifact_rag_root):
        output_resolved = output_dir.resolve()
        if (
            output_resolved != overlay_resolved
            and overlay_resolved not in output_resolved.parents
            and output_resolved not in overlay_resolved.parents
        ):
            add_root(output_dir)
    return include_roots


def _planned_metadata_overlay_base_rag_id(rag_current_json: Path, explicit_base_rag_id: str | None) -> str:
    if explicit_base_rag_id:
        return str(explicit_base_rag_id)
    current = read_json(rag_current_json)
    return str(current.get("rag_id") or "")


def _path_retained_in_roots(path: Path, roots: Sequence[Path]) -> bool:
    path_resolved = path.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if path_resolved == root_resolved or root_resolved in path_resolved.parents:
            return True
    return False


def _command_path(path: Path) -> str:
    text = str(path)
    return text.replace("\\", "/")


def _build_verified_row_intake_batch(
    *,
    discovery_output_dir: Path,
    output_dir: Path,
    rag_current_json: Path,
    default_source_lock_pack_json: Path,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        discovery_output_dir=str(discovery_output_dir),
        expected_acceleration_mode="cpu_audio_tensor_extraction",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="verified_row_intake_batch_no_model_inference",
    )
    ready_rows = [
        row
        for row in iter_jsonl(discovery_output_dir / "rows.jsonl")
        if row.get("ready_for_verified_row_intake") is True
    ]
    errors: list[str] = []
    source_jsonls: list[str] = []
    manifest_jsons: list[str] = []
    intake_outputs: list[dict[str, Any]] = []
    total_rows = 0
    total_blocked = 0
    for index, row in enumerate(ready_rows, start=1):
        intake_dir = output_dir / f"vr{index:03d}"
        source_lock_pack_json = Path(str(row.get("source_lock_pack_json") or default_source_lock_pack_json))
        append_progress(
            progress_path,
            "build_verified_rows",
            start_time=start_time,
            intake_dir=str(intake_dir),
            verified_corpus_manifest_json=str(row.get("verified_corpus_manifest_json") or ""),
            row_index=index,
        )
        intake_manifest = build_audio_quality_verified_rows(
            verified_corpus_manifest_json=Path(str(row.get("verified_corpus_manifest_json") or "")),
            output_dir=intake_dir,
            rag_current_json=rag_current_json,
            source_lock_pack_json=source_lock_pack_json,
        )
        manifest_json = str(intake_dir / "manifest.json")
        manifest_jsons.append(manifest_json)
        total_rows += int(intake_manifest.get("row_count") or 0)
        total_blocked += int(intake_manifest.get("blocked_row_count") or 0)
        intake_errors = [str(item) for item in intake_manifest.get("errors", [])]
        if intake_manifest.get("build_status") != "passed":
            errors.append("verified_row_intake_blocked")
        errors.extend(intake_errors)
        rows_jsonl = str((intake_manifest.get("outputs") or {}).get("rows_jsonl") or "")
        if rows_jsonl:
            intake_outputs.append(
                {
                    "manifest_json": manifest_json,
                    "rows_jsonl": rows_jsonl,
                    "build_status": str(intake_manifest.get("build_status") or ""),
                    "row_count": int(intake_manifest.get("row_count") or 0),
                    "blocked_row_count": int(intake_manifest.get("blocked_row_count") or 0),
                    "delivery_allowed": False,
                    "release_ready": False,
                }
            )
            if intake_manifest.get("build_status") == "passed" and int(intake_manifest.get("row_count") or 0) > 0:
                source_jsonls.append(rows_jsonl)
    if not ready_rows:
        errors.append("ready_verified_corpus_rows_missing")
    if errors:
        source_jsonls = []
    manifest = {
        "schema": VERIFIED_ROW_INTAKE_SCHEMA,
        "manifest_type": "audio_quality_verified_row_intake_batch",
        "build_status": "passed" if ready_rows and not errors else "blocked",
        "errors": sorted(set(errors)),
        "discovery_output_dir": str(discovery_output_dir),
        "rag_current_json": str(rag_current_json),
        "source_lock_pack_json": str(default_source_lock_pack_json),
        "ready_discovery_row_count": len(ready_rows),
        "row_count": total_rows,
        "blocked_row_count": total_blocked,
        "source_jsonls": source_jsonls,
        "manifest_jsons": manifest_jsons,
        "intake_outputs": intake_outputs,
        "outputs": {
            "manifest_json": str(output_dir / "manifest.json"),
            "progress_jsonl": str(progress_path),
        },
        "presentation_status": "audio_quality_verified_row_intake_batch_not_for_listen_pack",
        "training_started": False,
        "promotion_allowed": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "verified_discovery_only": True,
            "uses_build_audio_quality_verified_rows": True,
            "blocked_intake_excluded_from_dataset": True,
            "generated_stems_never_ground_truth": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "manifest.json", manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        build_status=manifest["build_status"],
        row_count=total_rows,
        blocked_row_count=total_blocked,
        source_jsonl_count=len(source_jsonls),
    )
    return manifest


def _build_learning_handoff(
    *,
    output_dir: Path,
    cycle_output_dir: Path,
    dataset_manifest: dict[str, Any] | None,
    model_output_dir: Path,
    validation_output_json: Path,
    audit_packet_output_dir: Path,
    holdout_scores_jsonl: Path | None,
    score_manifest_json: Path | None,
    training_device: str,
    training_epochs: int,
    training_hidden_size: int,
    training_min_rows: int,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        cycle_output_dir=str(cycle_output_dir),
        expected_acceleration_mode="external_only_cpu_handoff_planning",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="learning_handoff_no_training_or_inference",
    )
    blocked_reasons: list[str] = []
    waiting_reasons: list[str] = []
    dataset_manifest_json = str(cycle_output_dir / "dataset" / "manifest.json") if dataset_manifest is not None else ""
    dataset_dir = cycle_output_dir / "dataset"
    safe_training_rows = int(dataset_manifest.get("safe_training_row_count") or 0) if dataset_manifest is not None else 0
    if dataset_manifest is None:
        blocked_reasons.append("dataset_manifest_missing")
    else:
        if dataset_manifest.get("delivery_allowed") is True:
            blocked_reasons.append("dataset_manifest_delivery_allowed_true")
        if dataset_manifest.get("release_ready") is True:
            blocked_reasons.append("dataset_manifest_release_ready_true")
        if safe_training_rows < int(training_min_rows):
            blocked_reasons.append("safe_training_rows_below_minimum")
    if str(training_device).lower() != "cuda":
        blocked_reasons.append("gpu_training_required")

    model_manifest_json = model_output_dir / "manifest.json"
    model_file = model_output_dir / "model.pt"
    if not model_manifest_json.exists():
        waiting_reasons.append("model_manifest_pending")
    if not validation_output_json.exists():
        waiting_reasons.append("validation_json_pending")
    if holdout_scores_jsonl is None:
        waiting_reasons.append("holdout_scores_pending")
    elif not holdout_scores_jsonl.exists():
        waiting_reasons.append("holdout_scores_missing")
    if score_manifest_json is not None and not score_manifest_json.exists():
        waiting_reasons.append("score_manifest_missing")

    ready_for_training = not blocked_reasons
    ready_for_validation = model_manifest_json.exists() and model_file.exists()
    ready_for_audit_packet = ready_for_validation and validation_output_json.exists()
    train_command = []
    validation_command = []
    audit_packet_command = []
    if ready_for_training:
        train_command = [
            "python",
            "/app/tools/train_audio_quality_critic.py",
            "--dataset-dir",
            _command_path(dataset_dir),
            "--output-dir",
            _command_path(model_output_dir),
            "--device",
            str(training_device),
            "--epochs",
            str(int(training_epochs)),
            "--hidden-size",
            str(int(training_hidden_size)),
            "--min-train-rows",
            str(int(training_min_rows)),
        ]
        validation_command = [
            "python",
            "/app/tools/validate_audio_quality_critic.py",
            "--model-dir",
            _command_path(model_output_dir),
            "--output-json",
            _command_path(validation_output_json),
        ]
        audit_packet_command = [
            "python",
            "/app/tools/build_audio_quality_experiment_audit_packet.py",
            "--output-dir",
            _command_path(audit_packet_output_dir),
            "--model-manifest-json",
            _command_path(model_manifest_json),
            "--validation-json",
            _command_path(validation_output_json),
            "--dataset-manifest-json",
            _command_path(Path(dataset_manifest_json)),
        ]
        if score_manifest_json is not None:
            audit_packet_command.extend(["--score-manifest-json", _command_path(score_manifest_json)])
        if holdout_scores_jsonl is not None:
            audit_packet_command.extend(["--holdout-scores-jsonl", _command_path(holdout_scores_jsonl)])

    manifest = {
        "schema": LEARNING_HANDOFF_SCHEMA,
        "manifest_type": "audio_quality_learning_handoff",
        "handoff_status": "ready_for_training" if ready_for_training else "blocked",
        "blocked_reasons": sorted(set(blocked_reasons)),
        "waiting_reasons": sorted(set(waiting_reasons)),
        "cycle_output_dir": str(cycle_output_dir),
        "dataset_manifest_json": dataset_manifest_json,
        "safe_training_row_count": safe_training_rows,
        "training_device": str(training_device),
        "training_epochs": int(training_epochs),
        "training_hidden_size": int(training_hidden_size),
        "training_min_rows": int(training_min_rows),
        "model_output_dir": str(model_output_dir),
        "model_manifest_json": str(model_manifest_json),
        "validation_output_json": str(validation_output_json),
        "audit_packet_output_dir": str(audit_packet_output_dir),
        "holdout_scores_jsonl": str(holdout_scores_jsonl or ""),
        "score_manifest_json": str(score_manifest_json or ""),
        "ready_for_training": ready_for_training,
        "ready_for_validation": ready_for_validation,
        "ready_for_audit_packet": ready_for_audit_packet,
        "train_command": train_command,
        "validation_command": validation_command,
        "audit_packet_command": audit_packet_command,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "promotion_approved": False,
        "presentation_status": "audio_quality_learning_handoff_not_for_listen_pack",
        "safety": {
            "handoff_only": True,
            "training_not_started": True,
            "validation_not_run": True,
            "audit_packet_not_run": True,
            "runtime_authority_not_changed": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "commands.jsonl", [{"stage": "train", "command": train_command}, {"stage": "validate", "command": validation_command}, {"stage": "audit_packet", "command": audit_packet_command}])
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        handoff_status=manifest["handoff_status"],
        ready_for_training=ready_for_training,
        ready_for_validation=ready_for_validation,
        ready_for_audit_packet=ready_for_audit_packet,
    )
    return manifest


def run_audio_quality_learning_cycle(
    *,
    outcomes_root: Path,
    output_dir: Path,
    rag_current_json: Path,
    review_queue_manifest_json: Path | None = None,
    review_decisions_jsonl: Path | None = None,
    source_lock_output_dir: Path | None = None,
    verified_discovery_output_dir: Path | None = None,
    verified_recovery_output_dir: Path | None = None,
    verified_material_request_output_dir: Path | None = None,
    promotion_discovery_output_dir: Path | None = None,
    metadata_overlay_output_dir: Path | None = None,
    metadata_overlay_base_rag_id: str | None = None,
    metadata_overlay_update_current: bool = True,
    code_index_status: str = "",
    code_index_fallback: str = "",
    docker_service: str = "ml-worker",
    gpu_status: str = "not_checked",
    project_root: Path | None = None,
    training_handoff_output_dir: Path | None = None,
    training_model_output_dir: Path | None = None,
    validation_output_json: Path | None = None,
    audit_packet_output_dir: Path | None = None,
    holdout_scores_jsonl: Path | None = None,
    score_manifest_json: Path | None = None,
    training_device: str = "cuda",
    training_epochs: int = 16,
    training_hidden_size: int = 32,
    training_min_rows: int = 2,
) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    if (
        metadata_overlay_output_dir is not None
        and source_lock_output_dir is None
        and verified_discovery_output_dir is None
        and verified_recovery_output_dir is None
        and verified_material_request_output_dir is None
        and promotion_discovery_output_dir is None
        and training_handoff_output_dir is None
    ):
        raise ValueError("metadata_overlay_requires_metadata_root")
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        outcomes_root=str(outcomes_root),
        review_queue_manifest_json=str(review_queue_manifest_json or ""),
        review_decisions_jsonl=str(review_decisions_jsonl or ""),
        source_lock_output_dir=str(source_lock_output_dir or ""),
        verified_discovery_output_dir=str(verified_discovery_output_dir or ""),
        verified_recovery_output_dir=str(verified_recovery_output_dir or ""),
        verified_material_request_output_dir=str(verified_material_request_output_dir or ""),
        promotion_discovery_output_dir=str(promotion_discovery_output_dir or ""),
        metadata_overlay_output_dir=str(metadata_overlay_output_dir or ""),
        training_handoff_output_dir=str(training_handoff_output_dir or ""),
    )
    source_lock_manifest: dict[str, Any] | None = None
    if source_lock_output_dir is not None:
        append_progress(
            progress_path,
            "build_source_lock",
            start_time=start_time,
            source_lock_output_dir=str(source_lock_output_dir),
            expected_acceleration_mode="external_only_cpu_source_lock",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="source_lock_metadata_only",
        )
        source_lock_manifest = build_audio_quality_source_lock(
            rag_current_json=rag_current_json,
            output_dir=source_lock_output_dir,
            code_index_status=str(code_index_status),
            code_index_fallback=str(code_index_fallback),
            docker_service=str(docker_service),
            gpu_status=str(gpu_status),
            project_root=project_root,
        )
    verified_discovery_manifest: dict[str, Any] | None = None
    if verified_discovery_output_dir is not None:
        if source_lock_output_dir is None:
            raise ValueError("verified_discovery_requires_source_lock_output_dir")
        append_progress(
            progress_path,
            "discover_verified_corpora",
            start_time=start_time,
            verified_discovery_output_dir=str(verified_discovery_output_dir),
            source_lock_pack_json=str(source_lock_output_dir / "pack.json"),
            expected_acceleration_mode="external_only_cpu_rag_metadata_discovery",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="verified_corpus_discovery_no_model_inference",
        )
        verified_discovery_manifest = discover_audio_quality_verified_corpora(
            rag_current_json=rag_current_json,
            source_lock_pack_json=source_lock_output_dir / "pack.json",
            output_dir=verified_discovery_output_dir,
        )
    verified_recovery_manifest: dict[str, Any] | None = None
    if verified_recovery_output_dir is not None:
        append_progress(
            progress_path,
            "build_verified_corpus_recovery_plan",
            start_time=start_time,
            verified_recovery_output_dir=str(verified_recovery_output_dir),
            expected_acceleration_mode="external_only_cpu_rag_metadata_recovery_planning",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="verified_corpus_recovery_plan_no_model_inference",
        )
        verified_recovery_manifest = build_audio_quality_verified_corpus_recovery_plan(
            rag_current_json=rag_current_json,
            output_dir=verified_recovery_output_dir,
            project_root=project_root,
        )
    verified_material_request_manifest: dict[str, Any] | None = None
    if verified_material_request_output_dir is not None:
        if verified_recovery_output_dir is None or verified_recovery_manifest is None:
            raise ValueError("verified_material_request_requires_verified_recovery_output_dir")
        append_progress(
            progress_path,
            "build_verified_material_request_packet",
            start_time=start_time,
            verified_material_request_output_dir=str(verified_material_request_output_dir),
            verified_recovery_manifest_json=str(verified_recovery_output_dir / "manifest.json"),
            expected_acceleration_mode="external_only_cpu_verified_material_request_planning",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="verified_material_request_packet_metadata_only",
        )
        verified_material_request_manifest = build_audio_quality_verified_material_request_packet(
            recovery_manifest_json=verified_recovery_output_dir / "manifest.json",
            output_dir=verified_material_request_output_dir,
        )
    verified_row_intake_manifest: dict[str, Any] | None = None
    verified_row_source_jsonls: list[Path] = []
    if (
        verified_discovery_manifest is not None
        and verified_discovery_manifest.get("build_status") == "passed"
        and int(verified_discovery_manifest.get("ready_count") or 0) > 0
        and source_lock_output_dir is not None
    ):
        append_progress(
            progress_path,
            "build_verified_row_intake",
            start_time=start_time,
            verified_discovery_output_dir=str(verified_discovery_output_dir or ""),
            verified_row_intake_output_dir=str(output_dir / "vrows"),
            expected_acceleration_mode="cpu_audio_tensor_extraction",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="verified_row_intake_no_model_inference",
        )
        verified_row_intake_manifest = _build_verified_row_intake_batch(
            discovery_output_dir=verified_discovery_output_dir,
            output_dir=output_dir / "vrows",
            rag_current_json=rag_current_json,
            default_source_lock_pack_json=source_lock_output_dir / "pack.json",
        )
        if verified_row_intake_manifest.get("build_status") == "passed":
            verified_row_source_jsonls = [
                Path(str(path))
                for path in verified_row_intake_manifest.get("source_jsonls", [])
                if str(path)
            ]
    verified, review, raw_excluded = _collect_outcomes(outcomes_root)
    review_decision_ingest_manifest: dict[str, Any] | None = None
    if review_queue_manifest_json is not None:
        append_progress(
            progress_path,
            "ingest_review_decisions",
            start_time=start_time,
            review_queue_manifest_json=str(review_queue_manifest_json),
            review_decisions_jsonl=str(review_decisions_jsonl or ""),
        )
        review_decision_ingest_manifest = ingest_audio_quality_review_decisions(
            review_queue_manifest_json=review_queue_manifest_json,
            decisions_jsonl=review_decisions_jsonl,
            output_dir=output_dir / "decision_ingest",
        )
        decision_verified, decision_review, decision_raw_excluded = _collect_outcomes(output_dir / "decision_ingest")
        verified.extend(decision_verified)
        review.extend(decision_review)
        raw_excluded += decision_raw_excluded
    write_jsonl(output_dir / REVIEW_FILE, review)
    dataset_manifest: dict[str, Any] | None = None
    dataset_source_jsonls = list(verified_row_source_jsonls)
    verified_row_intake_training_row_count = (
        int(verified_row_intake_manifest.get("row_count") or 0)
        if verified_row_intake_manifest is not None and verified_row_intake_manifest.get("build_status") == "passed"
        else 0
    )
    if verified:
        verified_jsonl = output_dir / "verified.jsonl"
        write_jsonl(verified_jsonl, verified)
        dataset_source_jsonls.append(verified_jsonl)
    if dataset_source_jsonls:
        dataset_manifest = build_audio_quality_critic_dataset(
            source_jsonls=dataset_source_jsonls,
            output_dir=output_dir / "dataset",
            rag_current_json=rag_current_json,
        )
    promotion_discovery_manifest: dict[str, Any] | None = None
    metadata_overlay_manifest: dict[str, Any] | None = None
    if promotion_discovery_output_dir is not None:
        append_progress(
            progress_path,
            "discover_promotion_candidates",
            start_time=start_time,
            rag_current_json=str(rag_current_json),
            promotion_discovery_output_dir=str(promotion_discovery_output_dir),
            expected_acceleration_mode="external_only_cpu_rag_metadata_discovery",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="promotion_candidate_discovery_no_model_inference",
        )
        promotion_discovery_manifest = discover_audio_quality_promotion_candidates(
            rag_current_json=rag_current_json,
            output_dir=promotion_discovery_output_dir,
        )
    training_handoff_manifest: dict[str, Any] | None = None
    if training_handoff_output_dir is not None:
        append_progress(
            progress_path,
            "build_training_handoff",
            start_time=start_time,
            training_handoff_output_dir=str(training_handoff_output_dir),
            expected_acceleration_mode="external_only_cpu_handoff_planning",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="learning_handoff_no_training_or_inference",
        )
        training_handoff_manifest = _build_learning_handoff(
            output_dir=training_handoff_output_dir,
            cycle_output_dir=output_dir,
            dataset_manifest=dataset_manifest,
            model_output_dir=training_model_output_dir or (output_dir / "model"),
            validation_output_json=validation_output_json or (output_dir / "validation.json"),
            audit_packet_output_dir=audit_packet_output_dir or (output_dir / "audit"),
            holdout_scores_jsonl=holdout_scores_jsonl,
            score_manifest_json=score_manifest_json,
            training_device=training_device,
            training_epochs=training_epochs,
            training_hidden_size=training_hidden_size,
            training_min_rows=training_min_rows,
        )
    metadata_overlay_include_roots: list[Path] = []
    metadata_roots = [
        root
        for root in (
            source_lock_output_dir,
            verified_discovery_output_dir,
            verified_recovery_output_dir,
            verified_material_request_output_dir,
            promotion_discovery_output_dir,
            training_handoff_output_dir,
        )
        if root is not None
    ]
    if metadata_overlay_output_dir is not None and metadata_roots:
        metadata_overlay_include_roots = _metadata_overlay_include_roots(
            rag_current_json=rag_current_json,
            output_dir=output_dir,
            metadata_roots=metadata_roots,
            metadata_overlay_output_dir=metadata_overlay_output_dir,
        )
    planned_metadata_overlay_base_rag_id = (
        _planned_metadata_overlay_base_rag_id(rag_current_json, metadata_overlay_base_rag_id)
        if metadata_overlay_include_roots
        else ""
    )
    manifest = {
        "schema": LEARNING_CYCLE_SCHEMA,
        "manifest_type": "audio_quality_learning_cycle",
        "outcomes_root": str(outcomes_root),
        "review_queue_manifest_json": str(review_queue_manifest_json or ""),
        "review_decisions_jsonl": str(review_decisions_jsonl or ""),
        "source_lock_output_dir": str(source_lock_output_dir or ""),
        "source_lock_pack_json": (
            str(source_lock_output_dir / "pack.json")
            if source_lock_manifest is not None and source_lock_output_dir is not None
            else ""
        ),
        "source_lock_current_rag_id": (
            str(source_lock_manifest.get("current_rag_id") or "") if source_lock_manifest is not None else ""
        ),
        "source_lock_retained_in_rag": bool(
            source_lock_output_dir is not None
            and any(path.resolve() == source_lock_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        "verified_discovery_output_dir": str(verified_discovery_output_dir or ""),
        "verified_discovery_manifest_json": (
            str(verified_discovery_output_dir / "manifest.json")
            if verified_discovery_manifest is not None and verified_discovery_output_dir is not None
            else ""
        ),
        "verified_discovery_status": (
            str(verified_discovery_manifest.get("build_status") or "")
            if verified_discovery_manifest is not None
            else ""
        ),
        "verified_discovery_errors": (
            [str(item) for item in verified_discovery_manifest.get("errors", [])]
            if verified_discovery_manifest is not None
            else []
        ),
        "verified_discovery_ready_count": (
            int(verified_discovery_manifest.get("ready_count") or 0)
            if verified_discovery_manifest is not None
            else 0
        ),
        "verified_discovery_retained_in_rag": bool(
            verified_discovery_output_dir is not None
            and any(path.resolve() == verified_discovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        "verified_recovery_output_dir": str(verified_recovery_output_dir or ""),
        "verified_recovery_manifest_json": (
            str(verified_recovery_output_dir / "manifest.json")
            if verified_recovery_manifest is not None and verified_recovery_output_dir is not None
            else ""
        ),
        "verified_recovery_status": (
            str(verified_recovery_manifest.get("recovery_status") or "")
            if verified_recovery_manifest is not None
            else ""
        ),
        "verified_recovery_raw_reference_manifest_count": (
            int(verified_recovery_manifest.get("raw_reference_manifest_count") or 0)
            if verified_recovery_manifest is not None
            else 0
        ),
        "verified_recovery_retained_in_rag": bool(
            verified_recovery_output_dir is not None
            and any(path.resolve() == verified_recovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        "verified_material_request_output_dir": str(verified_material_request_output_dir or ""),
        "verified_material_request_manifest_json": (
            str(verified_material_request_output_dir / "manifest.json")
            if verified_material_request_manifest is not None and verified_material_request_output_dir is not None
            else ""
        ),
        "verified_material_request_status": (
            str(verified_material_request_manifest.get("request_status") or "")
            if verified_material_request_manifest is not None
            else ""
        ),
        "verified_material_request_count": (
            int(verified_material_request_manifest.get("request_count") or 0)
            if verified_material_request_manifest is not None
            else 0
        ),
        "verified_material_request_retained_in_rag": bool(
            verified_material_request_output_dir is not None
            and any(path.resolve() == verified_material_request_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        "verified_row_intake_output_dir": str(output_dir / "vrows" if verified_row_intake_manifest is not None else ""),
        "verified_row_intake_manifest_json": str(
            output_dir / "vrows" / "manifest.json" if verified_row_intake_manifest is not None else ""
        ),
        "verified_row_intake_status": (
            str(verified_row_intake_manifest.get("build_status") or "")
            if verified_row_intake_manifest is not None
            else ""
        ),
        "verified_row_intake_errors": (
            [str(item) for item in verified_row_intake_manifest.get("errors", [])]
            if verified_row_intake_manifest is not None
            else []
        ),
        "verified_row_intake_row_count": (
            int(verified_row_intake_manifest.get("row_count") or 0)
            if verified_row_intake_manifest is not None
            else 0
        ),
        "verified_row_intake_training_row_count": verified_row_intake_training_row_count,
        "verified_row_intake_blocked_row_count": (
            int(verified_row_intake_manifest.get("blocked_row_count") or 0)
            if verified_row_intake_manifest is not None
            else 0
        ),
        "verified_row_intake_source_jsonls": [str(path) for path in verified_row_source_jsonls],
        "verified_row_intake_retained_in_rag": bool(
            verified_row_intake_manifest is not None and _path_retained_in_roots(output_dir / "vrows", metadata_overlay_include_roots)
        ),
        "review_decision_ingest_manifest_json": (
            str(output_dir / "decision_ingest" / "manifest.json") if review_decision_ingest_manifest else ""
        ),
        "review_decision_ingest_status": (
            str(review_decision_ingest_manifest.get("ingest_status") or "") if review_decision_ingest_manifest else ""
        ),
        "review_decision_verified_outcome_count": (
            int(review_decision_ingest_manifest.get("verified_outcome_count") or 0)
            if review_decision_ingest_manifest
            else 0
        ),
        "review_decision_blocked_count": (
            int(review_decision_ingest_manifest.get("blocked_decision_count") or 0)
            if review_decision_ingest_manifest
            else 0
        ),
        "verified_examples_total": len(verified),
        "verified_training_rows_total": len(verified) + verified_row_intake_training_row_count,
        "review_queue_count": len(review),
        "raw_prediction_rows_excluded": raw_excluded,
        "dataset_manifest_json": str(output_dir / "dataset" / "manifest.json") if dataset_manifest else "",
        "promotion_discovery_output_dir": str(promotion_discovery_output_dir or ""),
        "promotion_discovery_manifest_json": (
            str(promotion_discovery_output_dir / "manifest.json")
            if promotion_discovery_manifest is not None and promotion_discovery_output_dir is not None
            else ""
        ),
        "promotion_discovery_status": (
            str(promotion_discovery_manifest.get("discovery_status") or "")
            if promotion_discovery_manifest is not None
            else ""
        ),
        "promotion_candidate_count": (
            int(promotion_discovery_manifest.get("candidate_count") or 0)
            if promotion_discovery_manifest is not None
            else 0
        ),
        "promotion_discovery_blocked_ref_count": (
            int(promotion_discovery_manifest.get("blocked_ref_count") or 0)
            if promotion_discovery_manifest is not None
            else 0
        ),
        "promotion_discovery_retained_in_rag": bool(
            promotion_discovery_output_dir is not None
            and any(path.resolve() == promotion_discovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        "training_handoff_output_dir": str(training_handoff_output_dir or ""),
        "training_handoff_manifest_json": (
            str(training_handoff_output_dir / "manifest.json")
            if training_handoff_manifest is not None and training_handoff_output_dir is not None
            else ""
        ),
        "training_handoff_status": (
            str(training_handoff_manifest.get("handoff_status") or "")
            if training_handoff_manifest is not None
            else ""
        ),
        "training_handoff_ready_for_training": (
            bool(training_handoff_manifest.get("ready_for_training"))
            if training_handoff_manifest is not None
            else False
        ),
        "training_handoff_retained_in_rag": bool(
            training_handoff_output_dir is not None
            and any(path.resolve() == training_handoff_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        "learning_cycle_retained_in_rag": any(path.resolve() == output_dir.resolve() for path in metadata_overlay_include_roots),
        "metadata_overlay_include_roots": [str(path) for path in metadata_overlay_include_roots],
        "metadata_overlay_output_dir": str(metadata_overlay_output_dir or ""),
        "metadata_overlay_manifest_json": (
            str(metadata_overlay_output_dir / "idx.json")
            if metadata_overlay_include_roots and metadata_overlay_output_dir is not None
            else ""
        ),
        "metadata_overlay_rag_id": (
            str(metadata_overlay_output_dir.name) if metadata_overlay_include_roots and metadata_overlay_output_dir is not None else ""
        ),
        "metadata_overlay_base_rag_id": planned_metadata_overlay_base_rag_id,
        "metadata_overlay_current_updated": bool(metadata_overlay_include_roots and metadata_overlay_update_current),
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "promotion_approved": False,
        "presentation_status": "audio_quality_learning_cycle_not_for_listen_pack",
        "safety": {
            "raw_predictions_excluded_from_training": True,
            "only_verified_outcomes_feed_dataset": True,
            "review_decisions_must_be_human_or_reference": True,
            "no_self_label_as_ground_truth": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "manifest.json", manifest)
    if metadata_overlay_include_roots and metadata_overlay_output_dir is not None:
        append_progress(
            progress_path,
            "retain_learning_cycle_metadata_in_rag",
            start_time=start_time,
            metadata_overlay_output_dir=str(metadata_overlay_output_dir),
            metadata_overlay_base_rag_id=planned_metadata_overlay_base_rag_id,
            metadata_overlay_update_current=bool(metadata_overlay_update_current),
            metadata_overlay_include_roots=[str(path) for path in metadata_overlay_include_roots],
            expected_acceleration_mode="cpu_filesystem_metadata_only",
            gpu_inference_skipped=True,
            gpu_inference_skip_reason="artifact_rag_metadata_overlay_no_model_inference",
        )
        metadata_overlay_manifest = build_artifact_rag_metadata_overlay(
            rag_current_json=rag_current_json,
            output_dir=metadata_overlay_output_dir,
            include_roots=metadata_overlay_include_roots,
            base_rag_id=metadata_overlay_base_rag_id,
            update_current=metadata_overlay_update_current,
        )
        manifest["metadata_overlay_rag_id"] = str(metadata_overlay_manifest.get("rag_id") or "")
        manifest["metadata_overlay_base_rag_id"] = str(metadata_overlay_manifest.get("base_rag_id") or "")
        manifest["metadata_overlay_current_updated"] = bool(metadata_overlay_update_current)
        manifest["source_lock_retained_in_rag"] = bool(
            source_lock_output_dir is not None
            and any(path.resolve() == source_lock_output_dir.resolve() for path in metadata_overlay_include_roots)
        )
        manifest["verified_discovery_retained_in_rag"] = bool(
            verified_discovery_output_dir is not None
            and any(path.resolve() == verified_discovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        )
        manifest["verified_recovery_retained_in_rag"] = bool(
            verified_recovery_output_dir is not None
            and any(path.resolve() == verified_recovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        )
        manifest["verified_material_request_retained_in_rag"] = bool(
            verified_material_request_output_dir is not None
            and any(path.resolve() == verified_material_request_output_dir.resolve() for path in metadata_overlay_include_roots)
        )
        manifest["verified_row_intake_retained_in_rag"] = bool(
            verified_row_intake_manifest is not None and _path_retained_in_roots(output_dir / "vrows", metadata_overlay_include_roots)
        )
        manifest["promotion_discovery_retained_in_rag"] = bool(
            promotion_discovery_output_dir is not None
            and any(path.resolve() == promotion_discovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        )
        manifest["training_handoff_retained_in_rag"] = bool(
            training_handoff_output_dir is not None
            and any(path.resolve() == training_handoff_output_dir.resolve() for path in metadata_overlay_include_roots)
        )
        manifest["learning_cycle_retained_in_rag"] = any(path.resolve() == output_dir.resolve() for path in metadata_overlay_include_roots)
        write_json(output_dir / "manifest.json", manifest)
    append_progress(
        progress_path,
        "finish",
        start_time=start_time,
        verified_examples_total=len(verified),
        review_queue_count=len(review),
        raw_prediction_rows_excluded=raw_excluded,
        source_lock_current_rag_id=(
            str(source_lock_manifest.get("current_rag_id") or "") if source_lock_manifest is not None else ""
        ),
        source_lock_retained_in_rag=bool(
            source_lock_output_dir is not None
            and any(path.resolve() == source_lock_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        verified_discovery_status=(
            str(verified_discovery_manifest.get("build_status") or "")
            if verified_discovery_manifest is not None
            else ""
        ),
        verified_discovery_ready_count=(
            int(verified_discovery_manifest.get("ready_count") or 0)
            if verified_discovery_manifest is not None
            else 0
        ),
        verified_discovery_retained_in_rag=bool(
            verified_discovery_output_dir is not None
            and any(path.resolve() == verified_discovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        verified_recovery_status=(
            str(verified_recovery_manifest.get("recovery_status") or "")
            if verified_recovery_manifest is not None
            else ""
        ),
        verified_recovery_raw_reference_manifest_count=(
            int(verified_recovery_manifest.get("raw_reference_manifest_count") or 0)
            if verified_recovery_manifest is not None
            else 0
        ),
        verified_recovery_retained_in_rag=bool(
            verified_recovery_output_dir is not None
            and any(path.resolve() == verified_recovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        verified_material_request_status=(
            str(verified_material_request_manifest.get("request_status") or "")
            if verified_material_request_manifest is not None
            else ""
        ),
        verified_material_request_count=(
            int(verified_material_request_manifest.get("request_count") or 0)
            if verified_material_request_manifest is not None
            else 0
        ),
        verified_material_request_retained_in_rag=bool(
            verified_material_request_output_dir is not None
            and any(path.resolve() == verified_material_request_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        verified_row_intake_status=(
            str(verified_row_intake_manifest.get("build_status") or "")
            if verified_row_intake_manifest is not None
            else ""
        ),
        verified_row_intake_row_count=(
            int(verified_row_intake_manifest.get("row_count") or 0)
            if verified_row_intake_manifest is not None
            else 0
        ),
        verified_row_intake_retained_in_rag=bool(
            verified_row_intake_manifest is not None and _path_retained_in_roots(output_dir / "vrows", metadata_overlay_include_roots)
        ),
        review_decision_verified_outcome_count=(
            int(review_decision_ingest_manifest.get("verified_outcome_count") or 0)
            if review_decision_ingest_manifest
            else 0
        ),
        review_decision_blocked_count=(
            int(review_decision_ingest_manifest.get("blocked_decision_count") or 0)
            if review_decision_ingest_manifest
            else 0
        ),
        promotion_discovery_status=(
            str(promotion_discovery_manifest.get("discovery_status") or "")
            if promotion_discovery_manifest is not None
            else ""
        ),
        promotion_candidate_count=(
            int(promotion_discovery_manifest.get("candidate_count") or 0)
            if promotion_discovery_manifest is not None
            else 0
        ),
        promotion_discovery_retained_in_rag=bool(
            promotion_discovery_output_dir is not None
            and any(path.resolve() == promotion_discovery_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        training_handoff_status=(
            str(training_handoff_manifest.get("handoff_status") or "")
            if training_handoff_manifest is not None
            else ""
        ),
        training_handoff_retained_in_rag=bool(
            training_handoff_output_dir is not None
            and any(path.resolve() == training_handoff_output_dir.resolve() for path in metadata_overlay_include_roots)
        ),
        metadata_overlay_rag_id=(
            str(metadata_overlay_manifest.get("rag_id") or "") if metadata_overlay_manifest is not None else ""
        ),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe active-learning dataset from audio quality outcomes.")
    parser.add_argument("--outcomes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--review-queue-manifest-json", type=Path)
    parser.add_argument("--review-decisions-jsonl", type=Path)
    parser.add_argument("--source-lock-output-dir", type=Path)
    parser.add_argument("--verified-discovery-output-dir", type=Path)
    parser.add_argument("--verified-recovery-output-dir", type=Path)
    parser.add_argument("--verified-material-request-output-dir", type=Path)
    parser.add_argument("--promotion-discovery-output-dir", type=Path)
    parser.add_argument("--metadata-overlay-output-dir", type=Path)
    parser.add_argument("--metadata-overlay-base-rag-id")
    parser.add_argument("--no-metadata-overlay-current", action="store_true")
    parser.add_argument("--code-index-status", default="")
    parser.add_argument("--code-index-fallback", default="")
    parser.add_argument("--docker-service", default="ml-worker")
    parser.add_argument("--gpu-status", default="not_checked")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--training-handoff-output-dir", type=Path)
    parser.add_argument("--training-model-output-dir", type=Path)
    parser.add_argument("--validation-output-json", type=Path)
    parser.add_argument("--audit-packet-output-dir", type=Path)
    parser.add_argument("--holdout-scores-jsonl", type=Path)
    parser.add_argument("--score-manifest-json", type=Path)
    parser.add_argument("--training-device", default="cuda")
    parser.add_argument("--training-epochs", type=int, default=16)
    parser.add_argument("--training-hidden-size", type=int, default=32)
    parser.add_argument("--training-min-rows", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="run_audio_quality_learning_cycle.py", entrypoint_path="/app/tools/run_audio_quality_learning_cycle.py")
    args = parse_args(argv)
    manifest = run_audio_quality_learning_cycle(
        outcomes_root=args.outcomes_root,
        output_dir=args.output_dir,
        rag_current_json=args.rag_current_json,
        review_queue_manifest_json=args.review_queue_manifest_json,
        review_decisions_jsonl=args.review_decisions_jsonl,
        source_lock_output_dir=args.source_lock_output_dir,
        verified_discovery_output_dir=args.verified_discovery_output_dir,
        verified_recovery_output_dir=args.verified_recovery_output_dir,
        verified_material_request_output_dir=args.verified_material_request_output_dir,
        promotion_discovery_output_dir=args.promotion_discovery_output_dir,
        metadata_overlay_output_dir=args.metadata_overlay_output_dir,
        metadata_overlay_base_rag_id=args.metadata_overlay_base_rag_id,
        metadata_overlay_update_current=not bool(args.no_metadata_overlay_current),
        code_index_status=str(args.code_index_status),
        code_index_fallback=str(args.code_index_fallback),
        docker_service=str(args.docker_service),
        gpu_status=str(args.gpu_status),
        project_root=args.project_root,
        training_handoff_output_dir=args.training_handoff_output_dir,
        training_model_output_dir=args.training_model_output_dir,
        validation_output_json=args.validation_output_json,
        audit_packet_output_dir=args.audit_packet_output_dir,
        holdout_scores_jsonl=args.holdout_scores_jsonl,
        score_manifest_json=args.score_manifest_json,
        training_device=str(args.training_device),
        training_epochs=int(args.training_epochs),
        training_hidden_size=int(args.training_hidden_size),
        training_min_rows=int(args.training_min_rows),
    )
    print(json.dumps({"manifest_json": str(args.output_dir / "manifest.json"), "verified_examples_total": manifest["verified_examples_total"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
