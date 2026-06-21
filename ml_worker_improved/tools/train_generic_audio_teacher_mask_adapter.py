#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from train_teacher_mask_separator import (  # noqa: E402
    TeacherMaskTrainConfig,
    _require_cuda_memory_available,
    _select_device,
    apply_teacher_mask,
    audio_summary,
    fit_audio_pair,
    read_audio,
    resample_audio,
    save_checkpoint,
    sha256_file,
    train_teacher_mask_dataset,
    validate_stem_name,
    write_flac,
)

PROGRESS_NAME = "generic_audio_teacher_mask_adapter_progress.jsonl"
RUN_ROWS_NAME = "generic_audio_finetune_run_rows.jsonl"
MANIFEST_NAME = "generic_audio_teacher_mask_adapter_manifest.json"
PSEUDO_TEACHER_SCHEMA = "generic_finetuning_pseudo_teacher_audio_training_example.v1"
START_TIME = time.monotonic()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, *, device: str, **fields: Any) -> None:
    uses_gpu = str(device) == "cuda"
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "gpu_training" if uses_gpu else "cpu_diagnostic_training",
        "gpu_training_required": uses_gpu,
        "gpu_training_skipped": not uses_gpu,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def resolve_existing_path(value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("missing path")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def row_training_group_id(row: dict[str, Any]) -> str:
    for key in ("training_group_id", "source_plan_track_id", "canonical_track_id", "track_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def row_source_candidate_family(row: dict[str, Any]) -> str:
    for key in ("source_candidate_family", "candidate_family", "family", "candidate_config_family"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def select_training_rows(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    exclude_holdout_track_ids: set[str],
    max_examples_per_track: int,
    source_candidate_family_filter: str = "",
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    track_counts: dict[str, int] = {}
    family_filter = str(source_candidate_family_filter or "").strip()
    for row in rows:
        if row.get("trainable") is False:
            continue
        if str(row.get("stem_type") or row.get("stem") or "") != stem:
            continue
        if family_filter and row_source_candidate_family(row) != family_filter:
            continue
        track_id = row_training_group_id(row)
        if not track_id or track_id in exclude_holdout_track_ids:
            continue
        if max_examples_per_track > 0 and track_counts.get(track_id, 0) >= max_examples_per_track:
            continue
        source_path = resolve_existing_path(row.get("source_audio_path"))
        target_path = resolve_existing_path(row.get("target_audio_path"))
        key = (track_id, str(source_path), str(target_path))
        if key in seen:
            continue
        seen.add(key)
        track_counts[track_id] = track_counts.get(track_id, 0) + 1
        selected.append({**row, "source_audio_path": str(source_path), "target_audio_path": str(target_path)})
    return selected


def pseudo_teacher_row_unsafe_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("schema") != PSEUDO_TEACHER_SCHEMA:
        reasons.append("unexpected_pseudo_teacher_schema")
    if row.get("pseudo_teacher") is not True:
        reasons.append("missing_pseudo_teacher_flag")
    if row.get("pseudo_teacher_targets_are_ground_truth") is True:
        reasons.append("pseudo_teacher_claims_ground_truth")
    if row.get("human_verified") is True or row.get("reference_verified") is True:
        reasons.append("pseudo_teacher_claims_verification")
    if row.get("release_ready") is True:
        reasons.append("pseudo_teacher_release_ready_true")
    if row.get("delivery_allowed") is True:
        reasons.append("pseudo_teacher_delivery_allowed_true")
    return reasons


def validate_pseudo_teacher_rows(rows: Sequence[dict[str, Any]]) -> None:
    unsafe: list[str] = []
    for index, row in enumerate(rows, start=1):
        reasons = pseudo_teacher_row_unsafe_reasons(row)
        if reasons:
            row_id = str(row.get("id") or f"row_{index}")
            unsafe.append(f"{row_id}: {', '.join(reasons)}")
    if unsafe:
        raise ValueError("unsafe pseudo-teacher rows: " + "; ".join(unsafe))


def _fingerprint_payload(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    config: TeacherMaskTrainConfig,
    max_frames_per_pair: int,
    max_examples_per_track: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "id": str(row.get("id") or row.get("example_id") or ""),
                "schema": str(row.get("schema") or ""),
                "track_id": str(row.get("track_id") or ""),
                "stem": stem,
                "source_audio_path": str(row.get("source_audio_path") or ""),
                "target_audio_path": str(row.get("target_audio_path") or ""),
                "pseudo_teacher": row.get("pseudo_teacher") is True,
            }
        )
    payload.append(
        {
            "config": {
                "n_fft": int(config.n_fft),
                "hop_length": int(config.hop_length),
                "hidden_size": int(config.hidden_size),
                "epochs": int(config.epochs),
                "batch_size": int(config.batch_size),
                "learning_rate": float(config.learning_rate),
                "mask_output_mode": str(config.mask_output_mode),
                "complex_mask_ceiling": float(config.complex_mask_ceiling),
                "activity_weight": float(config.activity_weight),
                "leakage_suppression_weight": float(config.leakage_suppression_weight),
                "magnitude_loss_weight": float(config.magnitude_loss_weight),
                "source_preservation_weight": float(config.source_preservation_weight),
                "active_frame_oversample": int(config.active_frame_oversample),
                "context_frame_radius": int(config.context_frame_radius),
                "source_feature_memory_neighbors": int(config.source_feature_memory_neighbors),
                "source_feature_memory_chunk_size": int(config.source_feature_memory_chunk_size),
                "source_feature_memory_projection_dim": int(config.source_feature_memory_projection_dim),
                "source_feature_memory_max_frames": int(config.source_feature_memory_max_frames),
                "min_cuda_free_memory_mb": int(config.min_cuda_free_memory_mb),
                "max_frames_per_pair": int(max_frames_per_pair),
                "max_examples_per_track": int(max_examples_per_track),
                "seed": int(config.seed),
            }
        }
    )
    return payload


def fingerprint_rows(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    config: TeacherMaskTrainConfig,
    max_frames_per_pair: int = 0,
    max_examples_per_track: int = 0,
) -> str:
    payload = _fingerprint_payload(
        rows,
        stem=stem,
        config=config,
        max_frames_per_pair=int(max_frames_per_pair),
        max_examples_per_track=int(max_examples_per_track),
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_training_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    progress_path: Path,
    device: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        source_path = resolve_existing_path(row["source_audio_path"])
        target_path = resolve_existing_path(row["target_audio_path"])
        source_audio, source_sample_rate = read_audio(source_path)
        target_audio, target_sample_rate = read_audio(target_path)
        target_resampled = False
        if target_sample_rate != source_sample_rate:
            target_audio = resample_audio(target_audio, target_sample_rate, source_sample_rate)
            target_resampled = True
        source_audio, target_audio = fit_audio_pair(source_audio, target_audio)
        pairs.append((source_audio, target_audio))
        summary = {
            "index": index,
            "id": str(row.get("id") or row.get("example_id") or ""),
            "track_id": str(row.get("track_id") or ""),
            "source_audio_path": str(source_path),
            "target_audio_path": str(target_path),
            "source_sample_rate": int(source_sample_rate),
            "target_original_sample_rate": int(target_sample_rate),
            "target_resampled_to_source": target_resampled,
            "frame_count": int(source_audio.shape[0]),
            "pseudo_teacher": row.get("pseudo_teacher") is True,
            "schema": str(row.get("schema") or ""),
        }
        summaries.append(summary)
        progress(progress_path, "audio_pair_loaded", device=device, **summary)
    return pairs, summaries


def train_generic_audio_teacher_mask_adapter(
    *,
    audio_training_examples_jsonl: Path,
    output_dir: Path,
    stem: str,
    scope: str,
    pseudo_teacher_examples_jsonl: Path | None = None,
    exclude_holdout_track_ids: set[str] | None = None,
    min_unique_tracks: int = 2,
    max_examples_per_track: int = 0,
    max_frames_per_pair: int = 0,
    n_fft: int = TeacherMaskTrainConfig.n_fft,
    hop_length: int = TeacherMaskTrainConfig.hop_length,
    hidden_size: int = TeacherMaskTrainConfig.hidden_size,
    epochs: int = TeacherMaskTrainConfig.epochs,
    batch_size: int = TeacherMaskTrainConfig.batch_size,
    learning_rate: float = TeacherMaskTrainConfig.learning_rate,
    mask_ceiling: float = TeacherMaskTrainConfig.mask_ceiling,
    mask_output_mode: str = TeacherMaskTrainConfig.mask_output_mode,
    complex_mask_ceiling: float = TeacherMaskTrainConfig.complex_mask_ceiling,
    activity_weight: float = TeacherMaskTrainConfig.activity_weight,
    leakage_suppression_weight: float = TeacherMaskTrainConfig.leakage_suppression_weight,
    magnitude_loss_weight: float = TeacherMaskTrainConfig.magnitude_loss_weight,
    source_preservation_weight: float = TeacherMaskTrainConfig.source_preservation_weight,
    active_frame_oversample: int = TeacherMaskTrainConfig.active_frame_oversample,
    context_frame_radius: int = TeacherMaskTrainConfig.context_frame_radius,
    source_feature_memory_neighbors: int = 0,
    source_feature_memory_chunk_size: int = TeacherMaskTrainConfig.source_feature_memory_chunk_size,
    source_feature_memory_projection_dim: int = TeacherMaskTrainConfig.source_feature_memory_projection_dim,
    source_feature_memory_max_frames: int = TeacherMaskTrainConfig.source_feature_memory_max_frames,
    min_cuda_free_memory_mb: int = TeacherMaskTrainConfig.min_cuda_free_memory_mb,
    max_peak: float = TeacherMaskTrainConfig.max_peak,
    source_candidate_family_filter: str = "",
    seed: int = TeacherMaskTrainConfig.seed,
    device: str = TeacherMaskTrainConfig.device,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    exclude_holdout_track_ids = set(exclude_holdout_track_ids or set())
    config = TeacherMaskTrainConfig(
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        hidden_size=int(hidden_size),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        mask_ceiling=float(mask_ceiling),
        mask_output_mode=str(mask_output_mode),
        complex_mask_ceiling=float(complex_mask_ceiling),
        activity_weight=float(activity_weight),
        leakage_suppression_weight=float(leakage_suppression_weight),
        magnitude_loss_weight=float(magnitude_loss_weight),
        source_preservation_weight=float(source_preservation_weight),
        active_frame_oversample=int(active_frame_oversample),
        context_frame_radius=int(context_frame_radius),
        time_feature_count=0,
        source_feature_memory_neighbors=int(source_feature_memory_neighbors),
        source_feature_memory_store_complex_masks=False,
        source_feature_memory_chunk_size=int(source_feature_memory_chunk_size),
        source_feature_memory_projection_dim=int(source_feature_memory_projection_dim),
        source_feature_memory_max_frames=int(source_feature_memory_max_frames),
        min_cuda_free_memory_mb=int(min_cuda_free_memory_mb),
        max_peak=float(max_peak),
        seed=int(seed),
        device=str(device),
    )
    rows = read_jsonl(audio_training_examples_jsonl)
    pseudo_rows = read_jsonl(pseudo_teacher_examples_jsonl) if pseudo_teacher_examples_jsonl is not None else []
    validate_pseudo_teacher_rows(pseudo_rows)
    selected_verified_rows = select_training_rows(
        rows,
        stem=stem,
        exclude_holdout_track_ids=exclude_holdout_track_ids,
        max_examples_per_track=int(max_examples_per_track),
        source_candidate_family_filter=str(source_candidate_family_filter or ""),
    )
    selected_pseudo_rows = select_training_rows(
        pseudo_rows,
        stem=stem,
        exclude_holdout_track_ids=exclude_holdout_track_ids,
        max_examples_per_track=int(max_examples_per_track),
        source_candidate_family_filter=str(source_candidate_family_filter or ""),
    )
    selected_rows = selected_verified_rows + selected_pseudo_rows
    training_track_ids = sorted({row_training_group_id(row) for row in selected_rows if row_training_group_id(row)})
    pseudo_teacher_track_ids = sorted(
        {row_training_group_id(row) for row in selected_pseudo_rows if row_training_group_id(row)}
    )
    training_source_candidate_families = sorted(
        {row_source_candidate_family(row) for row in selected_rows if row_source_candidate_family(row)}
    )
    progress(
        progress_path,
        "start",
        device=str(device),
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        pseudo_teacher_examples_jsonl=str(pseudo_teacher_examples_jsonl or ""),
        verified_input_row_count=len(rows),
        pseudo_teacher_input_row_count=len(pseudo_rows),
        stem=stem,
        scope=scope,
        selected_training_example_count=len(selected_rows),
        verified_training_example_count=len(selected_verified_rows),
        pseudo_teacher_training_example_count=len(selected_pseudo_rows),
        unique_training_track_count=len(training_track_ids),
        excluded_holdout_track_ids=sorted(exclude_holdout_track_ids),
        max_examples_per_track=int(max_examples_per_track),
        source_candidate_family_filter=str(source_candidate_family_filter or ""),
    )
    if len(training_track_ids) < int(min_unique_tracks):
        raise ValueError(f"insufficient unique tracks for shared adapter: {len(training_track_ids)} < {min_unique_tracks}")
    guard_device = _select_device(str(device))
    _require_cuda_memory_available(
        guard_device,
        min_free_memory_mb=int(config.min_cuda_free_memory_mb),
        progress_path=progress_path,
        progress_context={
            "stem": stem,
            "scope": scope,
            "device": str(device),
            "expected_acceleration_mode": "gpu_training" if str(device) == "cuda" else "cpu_diagnostic_training",
            "gpu_training_required": str(device) == "cuda",
            "gpu_training_skipped": str(device) != "cuda",
        },
        phase="generic_audio_shared_adapter_training",
    )
    pairs, example_summaries = load_training_pairs(selected_rows, progress_path=progress_path, device=str(device))
    adapter_fingerprint = fingerprint_rows(
        selected_rows,
        stem=stem,
        config=config,
        max_frames_per_pair=int(max_frames_per_pair),
        max_examples_per_track=int(max_examples_per_track),
    )
    adapter_id = f"{stem}_shared_teacher_mask_adapter_{adapter_fingerprint[:16]}"
    model, metadata, train_diagnostics = train_teacher_mask_dataset(
        pairs,
        config=config,
        progress_path=progress_path,
        progress_context={"adapter_id": adapter_id, "stem": stem, "scope": scope},
        max_frames_per_pair=int(max_frames_per_pair),
    )
    preview_audio, apply_diagnostics = apply_teacher_mask(pairs[0][0], model=model, metadata=metadata, config=config)
    preview_path = output_dir / f"01_shared_teacher_mask_training_preview_{stem}.flac"
    write_flac(preview_path, preview_audio, int(example_summaries[0]["source_sample_rate"]))
    checkpoint_path = output_dir / f"{stem}_shared_teacher_mask_adapter.pt"
    save_checkpoint(checkpoint_path, model=model, metadata=metadata, config=config, stem_name=stem)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    progress(
        progress_path,
        "artifacts_written",
        device=str(device),
        adapter_id=adapter_id,
        checkpoint_path=str(checkpoint_path),
        training_preview_path=str(preview_path),
    )
    run_rows_path = output_dir / RUN_ROWS_NAME
    run_row = {
        "scope": scope,
        "status": "executed",
        "example_id": adapter_id,
        "track_id": adapter_id,
        "training_track_ids": training_track_ids,
        "source_candidate_family_filter": str(source_candidate_family_filter or ""),
        "training_source_candidate_families": training_source_candidate_families,
        "verified_training_example_count": len(selected_verified_rows),
        "pseudo_teacher_training_example_count": len(selected_pseudo_rows),
        "pseudo_teacher_track_ids": pseudo_teacher_track_ids,
        "source_audio_path": str(selected_rows[0]["source_audio_path"]),
        "target_audio_path": str(selected_rows[0]["target_audio_path"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / MANIFEST_NAME),
        "output_stem_paths": [str(preview_path)],
        "device_used": float(train_diagnostics.get("device_used", 0.0)),
        "progress_first_event": "start",
        "progress_last_event": "finish",
        "delivery_allowed": False,
    }
    write_jsonl(run_rows_path, [run_row])
    manifest = {
        "schema": "generic_audio_teacher_mask_adapter_manifest.v1",
        "manifest_type": "generic_audio_teacher_mask_adapter",
        "adapter_id": adapter_id,
        "scope": scope,
        "stem": stem,
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "pseudo_teacher_examples_jsonl": str(pseudo_teacher_examples_jsonl or ""),
        "selected_training_example_count": len(selected_rows),
        "verified_training_example_count": len(selected_verified_rows),
        "pseudo_teacher_training_example_count": len(selected_pseudo_rows),
        "max_examples_per_track": int(max_examples_per_track),
        "max_frames_per_pair": int(max_frames_per_pair),
        "source_candidate_family_filter": str(source_candidate_family_filter or ""),
        "unique_training_track_count": len(training_track_ids),
        "training_track_ids": training_track_ids,
        "training_source_candidate_families": training_source_candidate_families,
        "pseudo_teacher_track_ids": pseudo_teacher_track_ids,
        "excluded_holdout_track_ids": sorted(exclude_holdout_track_ids),
        "adapter_fingerprint": adapter_fingerprint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "training_preview_path": str(preview_path),
        "training_preview": audio_summary(preview_path),
        "training_examples": example_summaries,
        "training": {
            "config": {
                "n_fft": int(config.n_fft),
                "hop_length": int(config.hop_length),
                "hidden_size": int(config.hidden_size),
                "epochs": int(config.epochs),
                "batch_size": int(config.batch_size),
                "learning_rate": float(config.learning_rate),
                "mask_ceiling": float(config.mask_ceiling),
                "mask_output_mode": str(config.mask_output_mode),
                "complex_mask_ceiling": float(config.complex_mask_ceiling),
                "activity_weight": float(config.activity_weight),
                "leakage_suppression_weight": float(config.leakage_suppression_weight),
                "magnitude_loss_weight": float(config.magnitude_loss_weight),
                "source_preservation_weight": float(config.source_preservation_weight),
                "source_feature_memory_neighbors": int(config.source_feature_memory_neighbors),
                "source_feature_memory_chunk_size": int(config.source_feature_memory_chunk_size),
                "source_feature_memory_projection_dim": int(config.source_feature_memory_projection_dim),
                "source_feature_memory_max_frames": int(config.source_feature_memory_max_frames),
                "min_cuda_free_memory_mb": int(config.min_cuda_free_memory_mb),
                "device": str(config.device),
            },
            "diagnostics": train_diagnostics,
            "application_preview_diagnostics": apply_diagnostics,
        },
        "contract": {
            "target_reference_used_at_training": True,
            "verified_target_reference_used_at_training": len(selected_verified_rows) > 0,
            "pseudo_teacher_targets_used_at_training": len(selected_pseudo_rows) > 0,
            "pseudo_teacher_targets_are_ground_truth": False,
            "dirty_stems_never_ground_truth": True,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "requires_holdout_validation": True,
            "holdout_validation_status": "missing",
            "delivery_allowed_without_reference": False,
            "per_track_hardcode_allowed": False,
            "stores_timestamp_rules": False,
            "stores_learned_source_feature_memory": int(config.source_feature_memory_neighbors) > 0,
            "source_candidate_family_specialist": bool(str(source_candidate_family_filter or "")),
            "source_feature_memory_generalization_claim": int(config.source_feature_memory_neighbors) > 0,
            "production_no_reference_generalization_claim": False,
        },
        "training_started": True,
        "fine_tune_started": True,
        "models_changed": True,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "training_artifact_not_for_listen_pack",
        "release_status": "training_artifact_only_requires_holdout_validation",
        "release_ready": False,
        "run_rows_jsonl": str(run_rows_path),
        "progress_jsonl": str(progress_path),
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        device=str(device),
        adapter_id=adapter_id,
        checkpoint_path=str(checkpoint_path),
        selected_training_example_count=len(selected_rows),
        unique_training_track_count=len(training_track_ids),
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one shared teacher-mask adapter from generic audio training examples.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--pseudo-teacher-examples-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--scope", default="")
    parser.add_argument("--exclude-holdout-track-id", action="append", default=[])
    parser.add_argument("--min-unique-tracks", type=int, default=2)
    parser.add_argument("--max-examples-per-track", type=int, default=0)
    parser.add_argument("--max-frames-per-pair", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=TeacherMaskTrainConfig.n_fft)
    parser.add_argument("--hop-length", type=int, default=TeacherMaskTrainConfig.hop_length)
    parser.add_argument("--hidden-size", type=int, default=TeacherMaskTrainConfig.hidden_size)
    parser.add_argument("--epochs", type=int, default=TeacherMaskTrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TeacherMaskTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TeacherMaskTrainConfig.learning_rate)
    parser.add_argument("--mask-ceiling", type=float, default=TeacherMaskTrainConfig.mask_ceiling)
    parser.add_argument("--mask-output-mode", default=TeacherMaskTrainConfig.mask_output_mode)
    parser.add_argument("--complex-mask-ceiling", type=float, default=TeacherMaskTrainConfig.complex_mask_ceiling)
    parser.add_argument("--activity-weight", type=float, default=TeacherMaskTrainConfig.activity_weight)
    parser.add_argument("--leakage-suppression-weight", type=float, default=TeacherMaskTrainConfig.leakage_suppression_weight)
    parser.add_argument("--magnitude-loss-weight", type=float, default=TeacherMaskTrainConfig.magnitude_loss_weight)
    parser.add_argument("--source-preservation-weight", type=float, default=TeacherMaskTrainConfig.source_preservation_weight)
    parser.add_argument("--active-frame-oversample", type=int, default=TeacherMaskTrainConfig.active_frame_oversample)
    parser.add_argument("--context-frame-radius", type=int, default=TeacherMaskTrainConfig.context_frame_radius)
    parser.add_argument("--source-feature-memory-neighbors", type=int, default=0)
    parser.add_argument("--source-feature-memory-chunk-size", type=int, default=TeacherMaskTrainConfig.source_feature_memory_chunk_size)
    parser.add_argument(
        "--source-feature-memory-projection-dim",
        type=int,
        default=TeacherMaskTrainConfig.source_feature_memory_projection_dim,
    )
    parser.add_argument(
        "--source-feature-memory-max-frames",
        type=int,
        default=TeacherMaskTrainConfig.source_feature_memory_max_frames,
    )
    parser.add_argument("--min-cuda-free-memory-mb", type=int, default=TeacherMaskTrainConfig.min_cuda_free_memory_mb)
    parser.add_argument("--max-peak", type=float, default=TeacherMaskTrainConfig.max_peak)
    parser.add_argument("--source-candidate-family-filter", default="")
    parser.add_argument("--seed", type=int, default=TeacherMaskTrainConfig.seed)
    parser.add_argument("--device", default=TeacherMaskTrainConfig.device)
    return parser


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="train_generic_audio_teacher_mask_adapter.py",
        entrypoint_path="/app/tools/train_generic_audio_teacher_mask_adapter.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_args = list(argv if argv is not None else sys.argv[1:])
    scope = str(args.scope or f"{args.stem}_shared_teacher_mask_adapter")
    manifest = train_generic_audio_teacher_mask_adapter(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        pseudo_teacher_examples_jsonl=args.pseudo_teacher_examples_jsonl,
        output_dir=args.output_dir,
        stem=str(args.stem),
        scope=scope,
        exclude_holdout_track_ids={str(item) for item in args.exclude_holdout_track_id},
        min_unique_tracks=int(args.min_unique_tracks),
        max_examples_per_track=int(args.max_examples_per_track),
        max_frames_per_pair=int(args.max_frames_per_pair),
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        hidden_size=int(args.hidden_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        mask_ceiling=float(args.mask_ceiling),
        mask_output_mode=str(args.mask_output_mode),
        complex_mask_ceiling=float(args.complex_mask_ceiling),
        activity_weight=float(args.activity_weight),
        leakage_suppression_weight=float(args.leakage_suppression_weight),
        magnitude_loss_weight=float(args.magnitude_loss_weight),
        source_preservation_weight=float(args.source_preservation_weight),
        active_frame_oversample=int(args.active_frame_oversample),
        context_frame_radius=int(args.context_frame_radius),
        source_feature_memory_neighbors=int(args.source_feature_memory_neighbors),
        source_feature_memory_chunk_size=int(args.source_feature_memory_chunk_size),
        source_feature_memory_projection_dim=int(args.source_feature_memory_projection_dim),
        source_feature_memory_max_frames=int(args.source_feature_memory_max_frames),
        min_cuda_free_memory_mb=int(args.min_cuda_free_memory_mb),
        max_peak=float(args.max_peak),
        source_candidate_family_filter=str(args.source_candidate_family_filter or ""),
        seed=int(args.seed),
        device=str(args.device),
        command_args=command_args,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / MANIFEST_NAME), "adapter_id": manifest["adapter_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
