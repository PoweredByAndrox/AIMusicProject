#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
    acceleration_mode,
    apply_teacher_mask_checkpoint_to_source,
    fit_audio_pair,
    is_accelerator_device,
    read_audio,
    resample_audio,
    validate_stem_name,
)

HOLDOUT_ROWS_NAME = "generic_audio_finetune_holdout_rows.jsonl"
HOLDOUT_MANIFEST_NAME = "generic_audio_finetune_holdout_manifest.json"
PROGRESS_NAME = "generic_audio_finetune_holdout_progress.jsonl"
EPSILON = 1.0e-12
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, *, device: str, **fields: Any) -> None:
    uses_gpu = is_accelerator_device(device)
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": acceleration_mode(device, phase="inference_holdout"),
        "gpu_inference_required": uses_gpu,
        "gpu_inference_skipped": not uses_gpu,
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


def _row_stem(row: dict[str, Any]) -> str:
    value = row.get("stem_type", row.get("stem", ""))
    return str(value or "").strip()


def _is_trainable_example(row: dict[str, Any], *, stem: str) -> bool:
    if row.get("trainable") is False:
        return False
    if _row_stem(row) != stem:
        return False
    return bool(str(row.get("source_audio_path") or "").strip() and str(row.get("target_audio_path") or "").strip())


def _row_source_candidate_family(row: dict[str, Any]) -> str:
    for key in ("source_candidate_family", "candidate_family", "family", "candidate_config_family"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def select_checkpoint_rows(
    rows: Sequence[dict[str, Any]],
    *,
    scope: str | None,
    require_gpu: bool,
    max_checkpoints: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if scope and str(row.get("scope") or "") != scope:
            continue
        if str(row.get("status") or "") == "failed":
            continue
        if str(row.get("progress_first_event") or "") != "start" or str(row.get("progress_last_event") or "") != "finish":
            continue
        if require_gpu and float(row.get("device_used", 0.0)) != 1.0:
            continue
        checkpoint_path = resolve_existing_path(row.get("checkpoint_path"))
        key = (str(row.get("checkpoint_sha256") or checkpoint_path), str(row.get("track_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        selected.append({**row, "checkpoint_path": str(checkpoint_path)})
        if max_checkpoints > 0 and len(selected) >= max_checkpoints:
            break
    return selected


def select_holdout_examples(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    forbidden_track_ids: set[str],
    max_holdouts: int,
    source_candidate_family_filter: str = "",
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    family_filter = str(source_candidate_family_filter or "").strip()
    for row in rows:
        if not _is_trainable_example(row, stem=stem):
            continue
        if family_filter and _row_source_candidate_family(row) != family_filter:
            continue
        holdout_track_id = str(row.get("track_id") or "")
        if not holdout_track_id or holdout_track_id in forbidden_track_ids:
            continue
        source_path = resolve_existing_path(row.get("source_audio_path"))
        target_path = resolve_existing_path(row.get("target_audio_path"))
        key = (holdout_track_id, str(source_path), str(target_path))
        if key in seen:
            continue
        seen.add(key)
        selected.append({**row, "source_audio_path": str(source_path), "target_audio_path": str(target_path)})
        if max_holdouts > 0 and len(selected) >= max_holdouts:
            break
    return selected


def checkpoint_training_track_ids(row: dict[str, Any]) -> set[str]:
    raw_ids = row.get("training_track_ids")
    ids: set[str] = set()
    if isinstance(raw_ids, list):
        ids.update(str(item) for item in raw_ids if str(item or "").strip())
    track_id = str(row.get("track_id") or "").strip()
    if track_id:
        ids.add(track_id)
    return ids


def _sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    residual = reference - estimate
    reference_energy = float(np.sum(np.square(reference), dtype=np.float64))
    residual_energy = float(np.sum(np.square(residual), dtype=np.float64))
    return float(10.0 * math.log10((reference_energy + EPSILON) / (residual_energy + EPSILON)))


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
    return float(20.0 * math.log10(max(rms, EPSILON)))


def load_aligned_audio_triplet(
    *,
    source_path: Path,
    target_path: Path,
    candidate_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_audio, source_rate = read_audio(source_path)
    target_audio, target_rate = read_audio(target_path)
    candidate_audio, candidate_rate = read_audio(candidate_path)
    alignment = {
        "source_sample_rate": int(source_rate),
        "target_original_sample_rate": int(target_rate),
        "candidate_original_sample_rate": int(candidate_rate),
        "target_resampled_to_source": False,
        "candidate_resampled_to_source": False,
    }
    if target_rate != source_rate:
        target_audio = resample_audio(target_audio, target_rate, source_rate)
        alignment["target_resampled_to_source"] = True
    if candidate_rate != source_rate:
        candidate_audio = resample_audio(candidate_audio, candidate_rate, source_rate)
        alignment["candidate_resampled_to_source"] = True
    source_audio, target_audio = fit_audio_pair(source_audio, target_audio)
    target_audio, candidate_audio = fit_audio_pair(target_audio, candidate_audio)
    source_audio, target_audio = fit_audio_pair(source_audio, target_audio)
    return source_audio, target_audio, candidate_audio, alignment


def _stem_output_path(manifest: dict[str, Any], stem: str, output_dir: Path) -> Path:
    stem_payload = (manifest.get("stems") or {}).get(stem)
    if isinstance(stem_payload, dict) and str(stem_payload.get("path") or "").strip():
        return resolve_existing_path(stem_payload["path"])
    fallback = output_dir / f"01_teacher_mask_checkpoint_{stem}.flac"
    if fallback.exists():
        return fallback
    raise ValueError(f"application manifest has no output path for stem={stem!r}")


def _failed_checks(*, candidate_sdr_db: float, improvement_db: float, min_sdr_db: float, min_improvement_db: float) -> list[str]:
    failed: list[str] = []
    if candidate_sdr_db < min_sdr_db:
        failed.append("sdr_below_threshold")
    if improvement_db < min_improvement_db:
        failed.append("improvement_below_threshold")
    return failed


def _application_failed_checks(error: Exception) -> list[str]:
    message = str(error)
    if "source_feature_memory_cost_guard" in message:
        return ["application_or_metric_error", "source_feature_memory_cost_guard"]
    return ["application_or_metric_error"]


def evaluate_holdout_pair(
    *,
    checkpoint_row: dict[str, Any],
    holdout_row: dict[str, Any],
    output_dir: Path,
    stem: str,
    device: str,
    min_sdr_db: float,
    min_improvement_db: float,
    progress_path: Path | None = None,
    progress_context: dict[str, Any] | None = None,
    max_source_feature_memory_distance_ops: int = DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
) -> dict[str, Any]:
    checkpoint_id = str(checkpoint_row.get("example_id") or "checkpoint")
    holdout_id = str(holdout_row.get("id") or holdout_row.get("example_id") or "holdout")
    pair_dir = output_dir / "applications" / f"{checkpoint_id}__to__{holdout_id}"
    manifest = apply_teacher_mask_checkpoint_to_source(
        source_mix_path=resolve_existing_path(holdout_row["source_audio_path"]),
        checkpoint_path=resolve_existing_path(checkpoint_row["checkpoint_path"]),
        output_dir=pair_dir,
        stem_name=stem,
        device=device,
        command_args=[],
        progress_path=progress_path,
        progress_context=progress_context,
        max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
    )
    application_diagnostics = manifest.get("application", {}).get("diagnostics", {})
    candidate_path = _stem_output_path(manifest, stem, pair_dir)
    source_audio, target_audio, candidate_audio, alignment = load_aligned_audio_triplet(
        source_path=resolve_existing_path(holdout_row["source_audio_path"]),
        target_path=resolve_existing_path(holdout_row["target_audio_path"]),
        candidate_path=candidate_path,
    )
    baseline_sdr_db = _sdr_db(target_audio, source_audio)
    candidate_sdr_db = _sdr_db(target_audio, candidate_audio)
    improvement_db = candidate_sdr_db - baseline_sdr_db
    failed_checks = _failed_checks(
        candidate_sdr_db=candidate_sdr_db,
        improvement_db=improvement_db,
        min_sdr_db=min_sdr_db,
        min_improvement_db=min_improvement_db,
    )
    return {
        "schema": "generic_audio_finetune_holdout_row.v1",
        "status": "passed" if not failed_checks else "failed",
        "failed_checks": failed_checks,
        "stem": stem,
        "checkpoint_example_id": checkpoint_id,
        "checkpoint_track_id": str(checkpoint_row.get("track_id") or ""),
        "checkpoint_path": str(checkpoint_row["checkpoint_path"]),
        "checkpoint_sha256": str(checkpoint_row.get("checkpoint_sha256") or ""),
        "holdout_example_id": holdout_id,
        "holdout_track_id": str(holdout_row.get("track_id") or ""),
        "holdout_source_audio_path": str(holdout_row["source_audio_path"]),
        "holdout_target_audio_path": str(holdout_row["target_audio_path"]),
        "holdout_source_candidate_family": _row_source_candidate_family(holdout_row),
        "candidate_audio_path": str(candidate_path),
        "application_manifest_path": str(pair_dir / "stem_manifest.json"),
        "application_device": str(application_diagnostics.get("device") or ""),
        "application_device_used": float(application_diagnostics.get("device_used", 0.0)),
        "metrics": {
            "baseline_sdr_db": round(baseline_sdr_db, 6),
            "candidate_sdr_db": round(candidate_sdr_db, 6),
            "improvement_db": round(improvement_db, 6),
            "target_rms_dbfs": round(_rms_dbfs(target_audio), 6),
            "candidate_rms_dbfs": round(_rms_dbfs(candidate_audio), 6),
            "compared_frame_count": int(target_audio.shape[0]),
        },
        "sample_rate_alignment": alignment,
        "thresholds": {
            "min_sdr_db": float(min_sdr_db),
            "min_improvement_db": float(min_improvement_db),
        },
        "delivery_allowed": False,
        "release_ready": False,
    }


def validate_generic_audio_finetune_holdout(
    *,
    run_rows_jsonl: Path,
    audio_training_examples_jsonl: Path,
    output_dir: Path,
    scope: str | None,
    stem: str,
    device: str = "cuda",
    require_gpu: bool = True,
    min_sdr_db: float = 3.0,
    min_improvement_db: float = 0.5,
    min_checkpoint_pass_rate: float = 1.0,
    max_checkpoints: int = 0,
    max_holdouts_per_checkpoint: int = 2,
    max_source_feature_memory_distance_ops: int = DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
    source_candidate_family_filter: str = "",
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    run_rows = read_jsonl(run_rows_jsonl)
    examples = read_jsonl(audio_training_examples_jsonl)
    checkpoint_rows = select_checkpoint_rows(
        run_rows,
        scope=scope,
        require_gpu=require_gpu,
        max_checkpoints=int(max_checkpoints),
    )
    progress(
        progress_path,
        "start",
        device=device,
        run_rows_jsonl=str(run_rows_jsonl),
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        scope=str(scope or ""),
        stem=stem,
        selected_checkpoint_count=len(checkpoint_rows),
        source_candidate_family_filter=str(source_candidate_family_filter or ""),
    )
    rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    failed_application_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_row in enumerate(checkpoint_rows, start=1):
        checkpoint_track_id = str(checkpoint_row.get("track_id") or "")
        forbidden_track_ids = checkpoint_training_track_ids(checkpoint_row)
        holdouts = select_holdout_examples(
            examples,
            stem=stem,
            forbidden_track_ids=forbidden_track_ids,
            max_holdouts=int(max_holdouts_per_checkpoint),
            source_candidate_family_filter=str(source_candidate_family_filter or ""),
        )
        progress(
            progress_path,
            "checkpoint_start",
            device=device,
            checkpoint_index=checkpoint_index,
            checkpoint_example_id=str(checkpoint_row.get("example_id") or ""),
            checkpoint_track_id=checkpoint_track_id,
            forbidden_track_ids=sorted(forbidden_track_ids),
            selected_holdout_count=len(holdouts),
        )
        checkpoint_rows_for_summary: list[dict[str, Any]] = []
        for holdout_index, holdout_row in enumerate(holdouts, start=1):
            progress(
                progress_path,
                "holdout_eval_start",
                device=device,
                checkpoint_index=checkpoint_index,
                holdout_index=holdout_index,
                checkpoint_example_id=str(checkpoint_row.get("example_id") or ""),
                holdout_example_id=str(holdout_row.get("id") or holdout_row.get("example_id") or ""),
                holdout_track_id=str(holdout_row.get("track_id") or ""),
                max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
            )
            holdout_progress_context = {
                "checkpoint_index": checkpoint_index,
                "holdout_index": holdout_index,
                "checkpoint_example_id": str(checkpoint_row.get("example_id") or ""),
                "holdout_example_id": str(holdout_row.get("id") or holdout_row.get("example_id") or ""),
                "holdout_track_id": str(holdout_row.get("track_id") or ""),
            }
            try:
                row = evaluate_holdout_pair(
                    checkpoint_row=checkpoint_row,
                    holdout_row=holdout_row,
                    output_dir=output_dir,
                    stem=stem,
                    device=device,
                    min_sdr_db=float(min_sdr_db),
                    min_improvement_db=float(min_improvement_db),
                    progress_path=progress_path,
                    progress_context=holdout_progress_context,
                    max_source_feature_memory_distance_ops=int(max_source_feature_memory_distance_ops),
                )
            except Exception as exc:
                failed_checks = _application_failed_checks(exc)
                progress(
                    progress_path,
                    "holdout_application_failed",
                    device=device,
                    checkpoint_index=checkpoint_index,
                    holdout_index=holdout_index,
                    failed_checks=failed_checks,
                    error=str(exc),
                )
                row = {
                    "schema": "generic_audio_finetune_holdout_row.v1",
                    "status": "failed",
                    "failed_checks": failed_checks,
                    "error": str(exc),
                    "stem": stem,
                    "checkpoint_example_id": str(checkpoint_row.get("example_id") or ""),
                    "checkpoint_track_id": checkpoint_track_id,
                    "checkpoint_path": str(checkpoint_row.get("checkpoint_path") or ""),
                    "holdout_example_id": str(holdout_row.get("id") or holdout_row.get("example_id") or ""),
                    "holdout_track_id": str(holdout_row.get("track_id") or ""),
                    "delivery_allowed": False,
                    "release_ready": False,
                }
                failed_application_rows.append(row)
            rows.append(row)
            checkpoint_rows_for_summary.append(row)
            progress(
                progress_path,
                "holdout_eval_finish",
                device=device,
                checkpoint_index=checkpoint_index,
                holdout_index=holdout_index,
                status=str(row.get("status") or ""),
                failed_checks=row.get("failed_checks") or [],
            )
        passed_count = sum(1 for row in checkpoint_rows_for_summary if row.get("status") == "passed")
        evaluated_count = len(checkpoint_rows_for_summary)
        pass_rate = float(passed_count / evaluated_count) if evaluated_count else 0.0
        checkpoint_status = "passed" if evaluated_count > 0 and pass_rate >= float(min_checkpoint_pass_rate) else "failed"
        checkpoint_summary = {
            "checkpoint_example_id": str(checkpoint_row.get("example_id") or ""),
            "checkpoint_track_id": checkpoint_track_id,
            "checkpoint_path": str(checkpoint_row.get("checkpoint_path") or ""),
            "holdout_eval_count": evaluated_count,
            "passed_holdout_count": passed_count,
            "failed_holdout_count": evaluated_count - passed_count,
            "pass_rate": round(pass_rate, 6),
            "status": checkpoint_status,
        }
        checkpoint_summaries.append(checkpoint_summary)
        progress(progress_path, "checkpoint_finish", device=device, **checkpoint_summary)
    rows_path = output_dir / HOLDOUT_ROWS_NAME
    write_jsonl(rows_path, rows)
    passed_eval_count = sum(1 for row in rows if row.get("status") == "passed")
    failed_eval_count = len(rows) - passed_eval_count
    checkpoint_pass_count = sum(1 for row in checkpoint_summaries if row.get("status") == "passed")
    holdout_status = (
        "passed"
        if checkpoint_rows and rows and checkpoint_pass_count == len(checkpoint_summaries) and failed_eval_count == 0
        else "failed"
    )
    manifest = {
        "schema": "generic_audio_finetune_holdout_manifest.v1",
        "manifest_type": "generic_audio_finetune_holdout",
        "run_rows_jsonl": str(run_rows_jsonl),
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "output_dir": str(output_dir),
        "scope": str(scope or ""),
        "stem": stem,
        "device": device,
        "require_gpu": bool(require_gpu),
        "holdout_mode": "cross_track_leave_one_checkpoint_out",
        "selected_checkpoint_count": len(checkpoint_rows),
        "checkpoint_pass_count": checkpoint_pass_count,
        "checkpoint_fail_count": len(checkpoint_summaries) - checkpoint_pass_count,
        "holdout_eval_count": len(rows),
        "passed_eval_count": passed_eval_count,
        "failed_eval_count": failed_eval_count,
        "failed_application_count": len(failed_application_rows),
        "checkpoint_summaries": checkpoint_summaries,
        "thresholds": {
            "min_sdr_db": float(min_sdr_db),
            "min_improvement_db": float(min_improvement_db),
            "min_checkpoint_pass_rate": float(min_checkpoint_pass_rate),
            "max_checkpoints": int(max_checkpoints),
            "max_holdouts_per_checkpoint": int(max_holdouts_per_checkpoint),
            "max_source_feature_memory_distance_ops": int(max_source_feature_memory_distance_ops),
            "source_candidate_family_filter": str(source_candidate_family_filter or ""),
        },
        "holdout_validation_status": holdout_status,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "holdout_training_artifact_not_for_listen_pack",
        "release_status": "blocked_until_holdout_and_promotion_gates",
        "release_ready": False,
        "holdout_rows_jsonl": str(rows_path),
        "progress_jsonl": str(progress_path),
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "command_args": list(command_args or []),
        },
    }
    manifest_path = output_dir / HOLDOUT_MANIFEST_NAME
    write_json(manifest_path, manifest)
    progress(
        progress_path,
        "finish",
        device=device,
        holdout_validation_status=holdout_status,
        selected_checkpoint_count=len(checkpoint_rows),
        holdout_eval_count=len(rows),
        failed_eval_count=failed_eval_count,
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generic audio fine-tune checkpoints on cross-track holdout examples.")
    parser.add_argument("--run-rows-jsonl", type=Path, required=True)
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument("--min-sdr-db", type=float, default=3.0)
    parser.add_argument("--min-improvement-db", type=float, default=0.5)
    parser.add_argument("--min-checkpoint-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--max-holdouts-per-checkpoint", type=int, default=2)
    parser.add_argument(
        "--max-source-feature-memory-distance-ops",
        type=int,
        default=DEFAULT_MAX_SOURCE_FEATURE_MEMORY_DISTANCE_OPS,
    )
    parser.add_argument("--source-candidate-family-filter", default="")
    parser.add_argument("--fail-on-failed-holdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="validate_generic_audio_finetune_holdout.py",
        entrypoint_path="/app/tools/validate_generic_audio_finetune_holdout.py",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_args = list(argv if argv is not None else sys.argv[1:])
    manifest = validate_generic_audio_finetune_holdout(
        run_rows_jsonl=args.run_rows_jsonl,
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        output_dir=args.output_dir,
        scope=args.scope,
        stem=args.stem,
        device=str(args.device),
        require_gpu=not bool(args.allow_non_gpu),
        min_sdr_db=float(args.min_sdr_db),
        min_improvement_db=float(args.min_improvement_db),
        min_checkpoint_pass_rate=float(args.min_checkpoint_pass_rate),
        max_checkpoints=int(args.max_checkpoints),
        max_holdouts_per_checkpoint=int(args.max_holdouts_per_checkpoint),
        max_source_feature_memory_distance_ops=int(args.max_source_feature_memory_distance_ops),
        source_candidate_family_filter=str(args.source_candidate_family_filter or ""),
        command_args=command_args,
    )
    print(json.dumps({"manifest_path": str(args.output_dir / HOLDOUT_MANIFEST_NAME), "holdout_validation_status": manifest["holdout_validation_status"]}, indent=2))
    if args.fail_on_failed_holdout and manifest["holdout_validation_status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
