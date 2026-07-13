from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime
from tools.oracle_stem_upper_bound import validate_stem_name
from tools.train_generic_audio_teacher_mask_adapter import (
    RUN_ROWS_NAME,
    train_generic_audio_teacher_mask_adapter,
)
from tools.train_teacher_mask_separator import (
    DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
    TeacherMaskTrainConfig,
    acceleration_mode,
    is_accelerator_device,
)
from tools.validate_generic_audio_finetune_holdout import (
    validate_generic_audio_finetune_holdout,
)


MATRIX_MANIFEST_NAME = "generic_audio_finetune_matrix_manifest.json"
MATRIX_ROWS_NAME = "generic_audio_finetune_matrix_rows.jsonl"
PROGRESS_NAME = "generic_audio_finetune_matrix_progress.jsonl"
CONFIG_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
START_TIME = time.monotonic()


INT_CONFIG_FIELDS = {
    "min_unique_tracks",
    "max_examples_per_track",
    "max_frames_per_pair",
    "n_fft",
    "hop_length",
    "hidden_size",
    "epochs",
    "batch_size",
    "active_frame_oversample",
    "context_frame_radius",
    "source_feature_memory_neighbors",
    "source_feature_memory_chunk_size",
    "source_feature_memory_projection_dim",
    "source_feature_memory_max_frames",
    "min_cuda_free_memory_mb",
    "seed",
}
FLOAT_CONFIG_FIELDS = {
    "learning_rate",
    "mask_ceiling",
    "complex_mask_ceiling",
    "activity_weight",
    "leakage_suppression_weight",
    "magnitude_loss_weight",
    "source_preservation_weight",
    "max_peak",
}
STR_CONFIG_FIELDS = {
    "mask_output_mode",
    "source_candidate_family_filter",
}
TRAIN_CONFIG_FIELDS = INT_CONFIG_FIELDS | FLOAT_CONFIG_FIELDS | STR_CONFIG_FIELDS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, *, device: str, require_gpu: bool, **fields: Any) -> None:
    uses_gpu = is_accelerator_device(device)
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": acceleration_mode(device, phase="training_and_holdout"),
        "gpu_training_required": bool(require_gpu),
        "gpu_inference_required": bool(require_gpu),
        "gpu_training_skipped": not uses_gpu,
        "gpu_inference_skipped": not uses_gpu,
    }
    if not uses_gpu:
        payload["gpu_skip_reason"] = "matrix_runner_device_not_accelerated"
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _release_cuda_between_matrix_phases(
    *,
    progress_path: Path,
    device: str,
    require_gpu: bool,
    cleanup_phase: str,
    config_index: int,
    config_id: str,
) -> None:
    progress(
        progress_path,
        "cuda_phase_cleanup_start",
        device=device,
        require_gpu=require_gpu,
        cleanup_phase=cleanup_phase,
        config_index=config_index,
        config_id=config_id,
    )
    normalized_device = str(device).strip().lower()
    if normalized_device not in {"cuda", "mps"}:
        progress(
            progress_path,
            "cuda_phase_cleanup_finish",
            device=device,
            require_gpu=require_gpu,
            cleanup_phase=cleanup_phase,
            config_index=config_index,
            config_id=config_id,
            cleanup_performed=False,
            cuda_available=False,
            skip_reason="device_not_accelerated",
        )
        return
    try:
        import torch

        gc.collect()
        cuda_available = normalized_device == "cuda" and bool(torch.cuda.is_available())
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = normalized_device == "mps" and mps_backend is not None and bool(mps_backend.is_available())
        if cuda_available:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_memory_mb = int(free_memory // (1024 * 1024))
            total_memory_mb = int(total_memory // (1024 * 1024))
        elif mps_available:
            torch.mps.synchronize()
            torch.mps.empty_cache()
            torch.mps.synchronize()
            free_memory_mb = None
            total_memory_mb = None
        else:
            free_memory_mb = None
            total_memory_mb = None
        gc.collect()
        progress(
            progress_path,
            "cuda_phase_cleanup_finish",
            device=device,
            require_gpu=require_gpu,
            cleanup_phase=cleanup_phase,
            config_index=config_index,
            config_id=config_id,
            cleanup_performed=cuda_available or mps_available,
            cuda_available=cuda_available,
            mps_available=mps_available,
            free_memory_mb=free_memory_mb,
            total_memory_mb=total_memory_mb,
        )
    except Exception as exc:
        progress(
            progress_path,
            "cuda_phase_cleanup_failed",
            device=device,
            require_gpu=require_gpu,
            cleanup_phase=cleanup_phase,
            config_index=config_index,
            config_id=config_id,
            error=str(exc),
        )
        if require_gpu:
            raise


def resolve_existing_file(path: Path, *, label: str) -> Path:
    resolved = path if path.is_absolute() else Path.cwd() / path
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _stable_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _sanitize_config_id(raw: str) -> str:
    value = CONFIG_ID_RE.sub("_", raw.strip()).strip("._-")
    if not value:
        raise ValueError("matrix config id is empty after sanitization")
    return value


def load_matrix_configs(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        raw_configs = payload.get("configs")
    else:
        raw_configs = payload
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("config matrix must be a non-empty list or an object with a non-empty configs list")
    configs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_config in enumerate(raw_configs, start=1):
        if not isinstance(raw_config, dict):
            raise ValueError(f"matrix config #{index} must be a JSON object")
        config = dict(raw_config)
        raw_id = str(config.get("id") or config.get("config_id") or f"config_{index:03d}_{_stable_config_hash(config)}")
        config_id = _sanitize_config_id(raw_id)
        if config_id in seen_ids:
            raise ValueError(f"duplicate matrix config id: {config_id}")
        seen_ids.add(config_id)
        config["id"] = config_id
        configs.append(config)
    return configs


def _int_config(config: dict[str, Any], field: str, default: int) -> int:
    return int(config.get(field, default))


def _float_config(config: dict[str, Any], field: str, default: float) -> float:
    return float(config.get(field, default))


def _str_config(config: dict[str, Any], field: str, default: str) -> str:
    return str(config.get(field, default))


def _unknown_config_fields(config: dict[str, Any]) -> list[str]:
    allowed = TRAIN_CONFIG_FIELDS | {"id", "config_id", "label", "notes"}
    return sorted(str(key) for key in config if str(key) not in allowed)


def _metrics_from_holdout_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidate_sdr = [
        float((row.get("metrics") or {}).get("candidate_sdr_db"))
        for row in rows
        if isinstance(row.get("metrics"), dict) and (row.get("metrics") or {}).get("candidate_sdr_db") is not None
    ]
    improvements = [
        float((row.get("metrics") or {}).get("improvement_db"))
        for row in rows
        if isinstance(row.get("metrics"), dict) and (row.get("metrics") or {}).get("improvement_db") is not None
    ]
    return {
        "min_candidate_sdr_db": round(min(candidate_sdr), 6) if candidate_sdr else None,
        "mean_candidate_sdr_db": round(sum(candidate_sdr) / len(candidate_sdr), 6) if candidate_sdr else None,
        "min_improvement_db": round(min(improvements), 6) if improvements else None,
        "mean_improvement_db": round(sum(improvements) / len(improvements), 6) if improvements else None,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_failed_check(failed_checks: list[str], check: str) -> None:
    if check not in failed_checks:
        failed_checks.append(check)


def _holdout_row_contract_failed_checks(
    *,
    holdout_manifest: dict[str, Any],
    holdout_rows: Sequence[dict[str, Any]],
    require_gpu: bool,
    min_sdr_db: float,
    min_improvement_db: float,
) -> list[str]:
    failed_checks: list[str] = []
    manifest_eval_count = int(holdout_manifest.get("holdout_eval_count") or 0)
    manifest_passed_count = int(holdout_manifest.get("passed_eval_count") or 0)
    manifest_failed_count = int(holdout_manifest.get("failed_eval_count") or 0)
    row_passed_count = sum(1 for row in holdout_rows if row.get("status") == "passed")
    row_failed_count = len(holdout_rows) - row_passed_count
    if manifest_eval_count != len(holdout_rows):
        _add_failed_check(failed_checks, "holdout_row_count_mismatch")
    if manifest_passed_count != row_passed_count or manifest_failed_count != row_failed_count:
        _add_failed_check(failed_checks, "holdout_row_status_count_mismatch")
    for row in holdout_rows:
        if row.get("status") != "passed":
            _add_failed_check(failed_checks, "holdout_row_failed")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        candidate_sdr = _optional_float(metrics.get("candidate_sdr_db"))
        improvement = _optional_float(metrics.get("improvement_db"))
        if candidate_sdr is None or improvement is None:
            _add_failed_check(failed_checks, "holdout_metrics_missing")
        elif candidate_sdr < float(min_sdr_db) or improvement < float(min_improvement_db):
            _add_failed_check(failed_checks, "holdout_metric_below_threshold")
        if require_gpu and _optional_float(row.get("application_device_used")) != 1.0:
            _add_failed_check(failed_checks, "holdout_application_not_gpu")
        if row.get("delivery_allowed") is True or row.get("release_ready") is True:
            _add_failed_check(failed_checks, "unsafe_holdout_row_delivery_contract")
    return failed_checks


def _score_tuple(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row.get("metrics") or {}

    def value(name: str) -> float:
        raw = metrics.get(name)
        return float(raw) if raw is not None else float("-inf")

    return (
        value("min_candidate_sdr_db"),
        value("mean_candidate_sdr_db"),
        value("min_improvement_db"),
        value("mean_improvement_db"),
    )


def _matrix_row_from_failure(*, config: dict[str, Any], config_scope: str, error: Exception, phase: str) -> dict[str, Any]:
    return {
        "schema": "generic_audio_finetune_matrix_row.v1",
        "config_id": str(config["id"]),
        "scope": config_scope,
        "status": "failed",
        "failed_phase": phase,
        "failed_checks": [f"{phase}_error"],
        "error": str(error),
        "train_config": {key: config[key] for key in sorted(config) if key != "id"},
        "delivery_allowed": False,
        "release_ready": False,
    }


def _nested_train_command_args(
    *,
    config_id: str,
    config_scope: str,
    audio_training_examples_jsonl: Path,
    pseudo_teacher_examples_jsonl: Path | None,
    train_dir: Path,
    stem: str,
    device: str,
    config: dict[str, Any],
) -> list[str]:
    args = [
        "--matrix-config-id",
        config_id,
        "--scope",
        config_scope,
        "--audio-training-examples-jsonl",
        str(audio_training_examples_jsonl),
        "--output-dir",
        str(train_dir),
        "--stem",
        stem,
        "--device",
        device,
    ]
    if pseudo_teacher_examples_jsonl is not None:
        args.extend(["--pseudo-teacher-examples-jsonl", str(pseudo_teacher_examples_jsonl)])
    for key in sorted(config):
        if key in {"id", "config_id", "label", "notes"}:
            continue
        args.extend([f"--{key.replace('_', '-')}", str(config[key])])
    return args


def _nested_holdout_command_args(
    *,
    config_id: str,
    config_scope: str,
    run_rows_jsonl: Path,
    audio_training_examples_jsonl: Path,
    holdout_dir: Path,
    stem: str,
    device: str,
    require_gpu: bool,
    min_sdr_db: float,
    min_improvement_db: float,
    min_checkpoint_pass_rate: float,
    max_holdouts_per_checkpoint: int,
    max_source_feature_memory_distance_ops: int,
    source_candidate_family_filter: str,
) -> list[str]:
    args = [
        "--matrix-config-id",
        config_id,
        "--scope",
        config_scope,
        "--run-rows-jsonl",
        str(run_rows_jsonl),
        "--audio-training-examples-jsonl",
        str(audio_training_examples_jsonl),
        "--output-dir",
        str(holdout_dir),
        "--stem",
        stem,
        "--device",
        device,
        "--min-sdr-db",
        str(min_sdr_db),
        "--min-improvement-db",
        str(min_improvement_db),
        "--min-checkpoint-pass-rate",
        str(min_checkpoint_pass_rate),
        "--max-holdouts-per-checkpoint",
        str(max_holdouts_per_checkpoint),
        "--max-source-feature-memory-distance-ops",
        str(max_source_feature_memory_distance_ops),
    ]
    if source_candidate_family_filter:
        args.extend(["--source-candidate-family-filter", str(source_candidate_family_filter)])
    if not require_gpu:
        args.append("--allow-non-gpu")
    return args


def run_generic_audio_finetune_matrix(
    *,
    audio_training_examples_jsonl: Path,
    config_matrix_json: Path,
    pseudo_teacher_examples_jsonl: Path | None = None,
    output_dir: Path,
    stem: str,
    scope: str,
    exclude_holdout_track_ids: set[str] | None = None,
    min_unique_tracks: int = 2,
    device: str = "cuda",
    require_gpu: bool = True,
    min_sdr_db: float = 3.0,
    min_improvement_db: float = 0.5,
    min_checkpoint_pass_rate: float = 1.0,
    max_holdouts_per_checkpoint: int = 2,
    max_source_feature_memory_distance_ops: int = DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    if require_gpu and not is_accelerator_device(device):
        raise ValueError("require_gpu=true requires --device cuda or --device mps for production matrix runs")
    audio_training_examples_jsonl = resolve_existing_file(audio_training_examples_jsonl, label="audio_training_examples_jsonl")
    if pseudo_teacher_examples_jsonl is not None:
        pseudo_teacher_examples_jsonl = resolve_existing_file(
            pseudo_teacher_examples_jsonl,
            label="pseudo_teacher_examples_jsonl",
        )
    config_matrix_json = resolve_existing_file(config_matrix_json, label="config_matrix_json")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    configs = load_matrix_configs(config_matrix_json)
    exclude_holdout_track_ids = set(exclude_holdout_track_ids or set())
    progress(
        progress_path,
        "start",
        device=device,
        require_gpu=require_gpu,
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        pseudo_teacher_examples_jsonl=str(pseudo_teacher_examples_jsonl or ""),
        config_matrix_json=str(config_matrix_json),
        stem=stem,
        scope=scope,
        config_count=len(configs),
        excluded_holdout_track_ids=sorted(exclude_holdout_track_ids),
    )
    rows: list[dict[str, Any]] = []
    training_attempt_count = 0
    successful_training_count = 0
    holdout_attempt_count = 0
    for config_index, config in enumerate(configs, start=1):
        config_id = str(config["id"])
        config_scope = f"{scope}__{config_id}"
        config_dir = output_dir / "configs" / f"{config_index:03d}_{config_id}"
        train_dir = config_dir / "train"
        holdout_dir = config_dir / "holdout"
        unknown_fields = _unknown_config_fields(config)
        progress(
            progress_path,
            "config_start",
            device=device,
            require_gpu=require_gpu,
            config_index=config_index,
            config_id=config_id,
            scope=config_scope,
        )
        try:
            if config_index > 1:
                _release_cuda_between_matrix_phases(
                    progress_path=progress_path,
                    device=device,
                    require_gpu=require_gpu,
                    cleanup_phase="before_next_config_training",
                    config_index=config_index,
                    config_id=config_id,
                )
            if unknown_fields:
                raise ValueError(f"unknown matrix config fields for {config_id}: {', '.join(unknown_fields)}")
            training_attempt_count += 1
            train_command_args = _nested_train_command_args(
                config_id=config_id,
                config_scope=config_scope,
                audio_training_examples_jsonl=audio_training_examples_jsonl,
                pseudo_teacher_examples_jsonl=pseudo_teacher_examples_jsonl,
                train_dir=train_dir,
                stem=stem,
                device=device,
                config=config,
            )
            train_manifest = train_generic_audio_teacher_mask_adapter(
                audio_training_examples_jsonl=audio_training_examples_jsonl,
                pseudo_teacher_examples_jsonl=pseudo_teacher_examples_jsonl,
                output_dir=train_dir,
                stem=stem,
                scope=config_scope,
                exclude_holdout_track_ids=exclude_holdout_track_ids,
                min_unique_tracks=_int_config(config, "min_unique_tracks", min_unique_tracks),
                max_examples_per_track=_int_config(config, "max_examples_per_track", 0),
                max_frames_per_pair=_int_config(config, "max_frames_per_pair", 0),
                n_fft=_int_config(config, "n_fft", TeacherMaskTrainConfig.n_fft),
                hop_length=_int_config(config, "hop_length", TeacherMaskTrainConfig.hop_length),
                hidden_size=_int_config(config, "hidden_size", TeacherMaskTrainConfig.hidden_size),
                epochs=_int_config(config, "epochs", TeacherMaskTrainConfig.epochs),
                batch_size=_int_config(config, "batch_size", TeacherMaskTrainConfig.batch_size),
                learning_rate=_float_config(config, "learning_rate", TeacherMaskTrainConfig.learning_rate),
                mask_ceiling=_float_config(config, "mask_ceiling", TeacherMaskTrainConfig.mask_ceiling),
                mask_output_mode=_str_config(config, "mask_output_mode", TeacherMaskTrainConfig.mask_output_mode),
                complex_mask_ceiling=_float_config(
                    config,
                    "complex_mask_ceiling",
                    TeacherMaskTrainConfig.complex_mask_ceiling,
                ),
                activity_weight=_float_config(config, "activity_weight", TeacherMaskTrainConfig.activity_weight),
                leakage_suppression_weight=_float_config(
                    config,
                    "leakage_suppression_weight",
                    TeacherMaskTrainConfig.leakage_suppression_weight,
                ),
                magnitude_loss_weight=_float_config(
                    config,
                    "magnitude_loss_weight",
                    TeacherMaskTrainConfig.magnitude_loss_weight,
                ),
                source_preservation_weight=_float_config(
                    config,
                    "source_preservation_weight",
                    TeacherMaskTrainConfig.source_preservation_weight,
                ),
                active_frame_oversample=_int_config(
                    config,
                    "active_frame_oversample",
                    TeacherMaskTrainConfig.active_frame_oversample,
                ),
                context_frame_radius=_int_config(
                    config,
                    "context_frame_radius",
                    TeacherMaskTrainConfig.context_frame_radius,
                ),
                source_feature_memory_neighbors=_int_config(config, "source_feature_memory_neighbors", 0),
                source_feature_memory_chunk_size=_int_config(
                    config,
                    "source_feature_memory_chunk_size",
                    TeacherMaskTrainConfig.source_feature_memory_chunk_size,
                ),
                source_feature_memory_projection_dim=_int_config(
                    config,
                    "source_feature_memory_projection_dim",
                    TeacherMaskTrainConfig.source_feature_memory_projection_dim,
                ),
                source_feature_memory_max_frames=_int_config(
                    config,
                    "source_feature_memory_max_frames",
                    TeacherMaskTrainConfig.source_feature_memory_max_frames,
                ),
                min_cuda_free_memory_mb=_int_config(
                    config,
                    "min_cuda_free_memory_mb",
                    TeacherMaskTrainConfig.min_cuda_free_memory_mb,
                ),
                max_peak=_float_config(config, "max_peak", TeacherMaskTrainConfig.max_peak),
                source_candidate_family_filter=_str_config(config, "source_candidate_family_filter", ""),
                seed=_int_config(config, "seed", TeacherMaskTrainConfig.seed),
                device=device,
                command_args=train_command_args,
            )
            successful_training_count += 1
            _release_cuda_between_matrix_phases(
                progress_path=progress_path,
                device=device,
                require_gpu=require_gpu,
                cleanup_phase="after_training_before_holdout",
                config_index=config_index,
                config_id=config_id,
            )
        except Exception as exc:
            row = _matrix_row_from_failure(config=config, config_scope=config_scope, error=exc, phase="training")
            rows.append(row)
            progress(
                progress_path,
                "config_finish",
                device=device,
                require_gpu=require_gpu,
                config_index=config_index,
                config_id=config_id,
                status="failed",
                failed_checks=row["failed_checks"],
            )
            continue

        try:
            run_rows_jsonl = Path(str(train_manifest.get("run_rows_jsonl") or train_dir / RUN_ROWS_NAME))
            holdout_attempt_count += 1
            holdout_command_args = _nested_holdout_command_args(
                config_id=config_id,
                config_scope=config_scope,
                run_rows_jsonl=run_rows_jsonl,
                audio_training_examples_jsonl=audio_training_examples_jsonl,
                holdout_dir=holdout_dir,
                stem=stem,
                device=device,
                require_gpu=require_gpu,
                min_sdr_db=min_sdr_db,
                min_improvement_db=min_improvement_db,
                min_checkpoint_pass_rate=min_checkpoint_pass_rate,
                max_holdouts_per_checkpoint=max_holdouts_per_checkpoint,
                max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
                source_candidate_family_filter=_str_config(config, "source_candidate_family_filter", ""),
            )
            holdout_manifest = validate_generic_audio_finetune_holdout(
                run_rows_jsonl=run_rows_jsonl,
                audio_training_examples_jsonl=audio_training_examples_jsonl,
                output_dir=holdout_dir,
                scope=config_scope,
                stem=stem,
                device=device,
                require_gpu=require_gpu,
                min_sdr_db=min_sdr_db,
                min_improvement_db=min_improvement_db,
                min_checkpoint_pass_rate=min_checkpoint_pass_rate,
                max_checkpoints=0,
                max_holdouts_per_checkpoint=max_holdouts_per_checkpoint,
                max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
                source_candidate_family_filter=_str_config(config, "source_candidate_family_filter", ""),
                command_args=holdout_command_args,
            )
            holdout_rows = read_jsonl(Path(str(holdout_manifest.get("holdout_rows_jsonl") or "")))
            metrics = _metrics_from_holdout_rows(holdout_rows)
            holdout_status = str(holdout_manifest.get("holdout_validation_status") or "failed")
            failed_checks: list[str] = []
            if train_manifest.get("delivery_allowed") is True or train_manifest.get("release_ready") is True:
                failed_checks.append("unsafe_training_delivery_contract")
            if holdout_status != "passed":
                failed_checks.append("holdout_failed")
            if not holdout_rows or int(holdout_manifest.get("holdout_eval_count") or 0) <= 0:
                failed_checks.append("holdout_rows_missing")
            if int(holdout_manifest.get("failed_eval_count") or 0) > 0:
                failed_checks.append("holdout_failed_evaluations")
            if holdout_manifest.get("delivery_allowed") is True or holdout_manifest.get("release_ready") is True:
                failed_checks.append("unsafe_holdout_delivery_contract")
            for check in _holdout_row_contract_failed_checks(
                holdout_manifest=holdout_manifest,
                holdout_rows=holdout_rows,
                require_gpu=require_gpu,
                min_sdr_db=min_sdr_db,
                min_improvement_db=min_improvement_db,
            ):
                _add_failed_check(failed_checks, check)
            row = {
                "schema": "generic_audio_finetune_matrix_row.v1",
                "config_id": config_id,
                "scope": config_scope,
                "status": "passed" if not failed_checks else "failed",
                "failed_checks": failed_checks,
                "train_config": {key: config[key] for key in sorted(config) if key != "id"},
                "adapter_manifest_path": str(train_dir / "generic_audio_teacher_mask_adapter_manifest.json"),
                "adapter_id": str(train_manifest.get("adapter_id") or ""),
                "checkpoint_path": str(train_manifest.get("checkpoint_path") or ""),
                "checkpoint_sha256": str(train_manifest.get("checkpoint_sha256") or ""),
                "run_rows_jsonl": str(run_rows_jsonl),
                "holdout_manifest_path": str(holdout_dir / "generic_audio_finetune_holdout_manifest.json"),
                "holdout_rows_jsonl": str(holdout_manifest.get("holdout_rows_jsonl") or ""),
                "holdout_validation_status": holdout_status,
                "holdout_eval_count": int(holdout_manifest.get("holdout_eval_count") or 0),
                "passed_eval_count": int(holdout_manifest.get("passed_eval_count") or 0),
                "failed_eval_count": int(holdout_manifest.get("failed_eval_count") or 0),
                "metrics": metrics,
                "delivery_allowed": False,
                "release_ready": False,
            }
            _release_cuda_between_matrix_phases(
                progress_path=progress_path,
                device=device,
                require_gpu=require_gpu,
                cleanup_phase="after_holdout_before_next_config",
                config_index=config_index,
                config_id=config_id,
            )
        except Exception as exc:
            row = _matrix_row_from_failure(config=config, config_scope=config_scope, error=exc, phase="holdout")
            _release_cuda_between_matrix_phases(
                progress_path=progress_path,
                device=device,
                require_gpu=require_gpu,
                cleanup_phase="after_holdout_failure_before_next_config",
                config_index=config_index,
                config_id=config_id,
            )
        rows.append(row)
        progress(
            progress_path,
            "config_finish",
            device=device,
            require_gpu=require_gpu,
            config_index=config_index,
            config_id=config_id,
            status=str(row.get("status") or ""),
            failed_checks=row.get("failed_checks") or [],
        )
    rows_path = output_dir / MATRIX_ROWS_NAME
    write_jsonl(rows_path, rows)
    passed_rows = [row for row in rows if row.get("status") == "passed"]
    best_row = max(passed_rows, key=_score_tuple) if passed_rows else None
    matrix_status = "passed" if best_row else "failed"
    manifest = {
        "schema": "generic_audio_finetune_matrix_manifest.v1",
        "manifest_type": "generic_audio_finetune_matrix",
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "pseudo_teacher_examples_jsonl": str(pseudo_teacher_examples_jsonl or ""),
        "config_matrix_json": str(config_matrix_json),
        "output_dir": str(output_dir),
        "stem": stem,
        "scope": scope,
        "device": device,
        "require_gpu": bool(require_gpu),
        "config_count": len(configs),
        "training_attempt_count": training_attempt_count,
        "successful_training_count": successful_training_count,
        "holdout_attempt_count": holdout_attempt_count,
        "passed_config_count": len(passed_rows),
        "failed_config_count": len(rows) - len(passed_rows),
        "matrix_validation_status": matrix_status,
        "best_config_id": str(best_row.get("config_id") or "") if best_row else "",
        "best_config": best_row,
        "thresholds": {
            "min_sdr_db": float(min_sdr_db),
            "min_improvement_db": float(min_improvement_db),
            "min_checkpoint_pass_rate": float(min_checkpoint_pass_rate),
            "max_holdouts_per_checkpoint": int(max_holdouts_per_checkpoint),
            "max_source_feature_memory_distance_ops": int(max_source_feature_memory_distance_ops),
        },
        "training_started": training_attempt_count > 0,
        "fine_tune_started": training_attempt_count > 0,
        "models_changed": successful_training_count > 0,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "training_matrix_artifact_not_for_listen_pack",
        "release_status": "blocked_until_holdout_and_promotion_gates",
        "release_ready": False,
        "contract": {
            "pseudo_teacher_rows_allowed_for_training": pseudo_teacher_examples_jsonl is not None,
            "pseudo_teacher_rows_not_used_for_holdout": True,
            "holdout_uses_verified_examples_only": True,
            "dirty_stems_never_ground_truth": True,
        },
        "matrix_rows_jsonl": str(rows_path),
        "progress_jsonl": str(progress_path),
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    manifest_path = output_dir / MATRIX_MANIFEST_NAME
    write_json(manifest_path, manifest)
    progress(
        progress_path,
        "finish",
        device=device,
        require_gpu=require_gpu,
        matrix_validation_status=matrix_status,
        passed_config_count=len(passed_rows),
        failed_config_count=len(rows) - len(passed_rows),
        best_config_id=manifest["best_config_id"],
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible shared teacher-mask adapter config matrix and holdout gate."
    )
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--pseudo-teacher-examples-jsonl", type=Path)
    parser.add_argument("--config-matrix-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--scope", default="")
    parser.add_argument("--exclude-holdout-track-id", action="append", default=[])
    parser.add_argument("--min-unique-tracks", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument("--min-sdr-db", type=float, default=3.0)
    parser.add_argument("--min-improvement-db", type=float, default=0.5)
    parser.add_argument("--min-checkpoint-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-holdouts-per-checkpoint", type=int, default=2)
    parser.add_argument(
        "--max-source-feature-memory-distance-ops",
        type=int,
        default=DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
    )
    parser.add_argument("--fail-on-no-passing-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_generic_audio_finetune_matrix.py",
        entrypoint_path="/app/tools/run_generic_audio_finetune_matrix.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_args = list(argv if argv is not None else sys.argv[1:])
    scope = str(args.scope or f"{args.stem}_shared_teacher_mask_adapter_matrix")
    manifest = run_generic_audio_finetune_matrix(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        pseudo_teacher_examples_jsonl=args.pseudo_teacher_examples_jsonl,
        config_matrix_json=args.config_matrix_json,
        output_dir=args.output_dir,
        stem=str(args.stem),
        scope=scope,
        exclude_holdout_track_ids={str(item) for item in args.exclude_holdout_track_id},
        min_unique_tracks=int(args.min_unique_tracks),
        device=str(args.device),
        require_gpu=not bool(args.allow_non_gpu),
        min_sdr_db=float(args.min_sdr_db),
        min_improvement_db=float(args.min_improvement_db),
        min_checkpoint_pass_rate=float(args.min_checkpoint_pass_rate),
        max_holdouts_per_checkpoint=int(args.max_holdouts_per_checkpoint),
        max_source_feature_memory_distance_ops=int(args.max_source_feature_memory_distance_ops),
        command_args=command_args,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MATRIX_MANIFEST_NAME),
                "matrix_validation_status": manifest["matrix_validation_status"],
                "best_config_id": manifest["best_config_id"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.fail_on_no_passing_config and manifest["matrix_validation_status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
