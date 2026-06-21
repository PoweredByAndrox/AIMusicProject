#!/usr/bin/env python3
"""Train and validate no-reference candidate-family blend adapters.

The adapter learns convex blend weights from verified candidate-conditioned
audio pairs. Holdout scoring compares the blend against the best individual
candidate family for the same holdout example, so a passing row must improve
the available source candidates rather than merely pick a weaker baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from docker_runtime import require_docker_runtime  # noqa: E402
from train_teacher_mask_separator import (  # noqa: E402
    fit_audio_pair,
    read_audio,
    resample_audio,
    sha256_file,
    validate_stem_name,
    write_flac,
)


START_TIME = time.monotonic()
MATRIX_MANIFEST_NAME = "generic_audio_finetune_matrix_manifest.json"
MATRIX_ROWS_NAME = "generic_audio_finetune_matrix_rows.jsonl"
PROGRESS_NAME = "candidate_family_blend_matrix_progress.jsonl"
RUN_ROWS_NAME = "candidate_family_blend_run_rows.jsonl"
HOLDOUT_ROWS_NAME = "candidate_family_blend_holdout_rows.jsonl"
CHECKPOINT_NAME = "candidate_family_blend_adapter.json"
EPSILON = 1.0e-8


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid jsonl row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object row")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_dsp_training_and_eval",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "candidate_family_blend_is_cpu_dsp_weight_search",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "candidate_family_blend_has_no_neural_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def resolve_existing_path(value: Any) -> Path:
    path = Path(str(value or ""))
    if not str(path):
        raise ValueError("missing audio path")
    if not path.exists():
        raise FileNotFoundError(f"audio path not found: {path}")
    return path


def row_is_verified(row: dict[str, Any]) -> bool:
    return row.get("reference_verified") is True or row.get("human_verified") is True


def row_family(row: dict[str, Any]) -> str:
    return str(row.get("source_candidate_family") or "").strip()


def row_group_key(row: dict[str, Any]) -> tuple[str, str]:
    track_id = str(row.get("track_id") or "").strip()
    target_key = str(row.get("target_audio_sha256") or row.get("target_audio_path") or "").strip()
    if not track_id:
        raise ValueError(f"training row is missing track_id: {row.get('id')!r}")
    if not target_key:
        raise ValueError(f"training row is missing target key: {row.get('id')!r}")
    return track_id, target_key


def _safe_group_id(group_key: tuple[str, str]) -> str:
    return stable_hash({"track_id": group_key[0], "target_key": group_key[1]})[:16]


def _row_allowed(row: dict[str, Any], *, stem: str) -> bool:
    if row.get("trainable") is False:
        return False
    if row.get("release_ready") is True:
        return False
    if str(row.get("stem_type") or row.get("stem") or "") != stem:
        return False
    return row_is_verified(row)


def complete_candidate_groups(
    rows: Sequence[dict[str, Any]],
    *,
    stem: str,
    families: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_set = {str(family) for family in families}
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    skipped_rows: list[dict[str, Any]] = []
    for row in rows:
        if not _row_allowed(row, stem=stem):
            skipped_rows.append({"id": str(row.get("id") or ""), "reason": "row_not_verified_trainable_stem_match"})
            continue
        family = row_family(row)
        if family not in family_set:
            continue
        key = row_group_key(row)
        grouped.setdefault(key, {})[family] = row

    groups: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for key, by_family in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        missing = [family for family in families if family not in by_family]
        if missing:
            incomplete.append({"track_id": key[0], "target_key": key[1], "missing_families": missing})
            continue
        groups.append(
            {
                "group_id": _safe_group_id(key),
                "track_id": key[0],
                "target_key": key[1],
                "rows_by_family": {family: by_family[family] for family in families},
            }
        )
    return groups, incomplete + skipped_rows


def load_group_audio(group: dict[str, Any], families: Sequence[str]) -> tuple[list[np.ndarray], np.ndarray, int]:
    family_rows = group["rows_by_family"]
    sources: list[np.ndarray] = []
    sample_rate: int | None = None
    for family in families:
        audio, rate = read_audio(resolve_existing_path(family_rows[family]["source_audio_path"]))
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            audio = resample_audio(audio, rate, sample_rate)
        sources.append(audio)
    if sample_rate is None:
        raise ValueError("candidate group has no source audio")
    target, target_rate = read_audio(resolve_existing_path(family_rows[families[0]]["target_audio_path"]))
    if target_rate != sample_rate:
        target = resample_audio(target, target_rate, sample_rate)
    for index, source in enumerate(list(sources)):
        source, target = fit_audio_pair(source, target)
        sources[index] = source
    min_length = min([source.shape[0] for source in sources] + [target.shape[0]])
    min_channels = min([source.shape[1] for source in sources] + [target.shape[1]])
    return [source[:min_length, :min_channels] for source in sources], target[:min_length, :min_channels], sample_rate


def sdr_db(target: np.ndarray, candidate: np.ndarray) -> float:
    target, candidate = fit_audio_pair(target, candidate)
    signal = float(np.sum(np.square(target), dtype=np.float64)) + EPSILON
    noise = float(np.sum(np.square(target - candidate), dtype=np.float64)) + EPSILON
    return 10.0 * math.log10(signal / noise)


def blend_audio(sources: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(sources) != len(weights):
        raise ValueError("source/weight count mismatch")
    candidate = np.zeros_like(sources[0], dtype=np.float64)
    for source, weight in zip(sources, weights):
        candidate = candidate + source.astype(np.float64, copy=False) * float(weight)
    return candidate


def simplex_weights(count: int, step: float) -> list[list[float]]:
    if count < 2:
        raise ValueError("at least two candidate families are required")
    if step <= 0.0 or step > 1.0:
        raise ValueError("weight_grid_step must be in (0, 1]")
    units = int(round(1.0 / float(step)))
    if units <= 0 or not math.isclose(units * float(step), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError("weight_grid_step must divide 1.0 evenly")

    weights: list[list[float]] = []

    def visit(prefix: list[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            weights.append([float(item) / float(units) for item in [*prefix, remaining]])
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots_left - 1)

    visit([], units, count)
    return weights


def train_global_weights(
    training_groups: Sequence[dict[str, Any]],
    *,
    families: Sequence[str],
    weight_grid_step: float,
    progress_path: Path,
    config_id: str,
) -> tuple[list[float], dict[str, Any]]:
    candidate_weights = simplex_weights(len(families), weight_grid_step)
    loaded: list[tuple[list[np.ndarray], np.ndarray]] = []
    for index, group in enumerate(training_groups, start=1):
        progress(progress_path, "train_group_load_start", config_id=config_id, group_index=index, group_id=group["group_id"])
        sources, target, _sample_rate = load_group_audio(group, families)
        loaded.append((sources, target))
        progress(progress_path, "train_group_load_finish", config_id=config_id, group_index=index, group_id=group["group_id"])

    best_weights: list[float] | None = None
    best_mean_sdr = -float("inf")
    for weights in candidate_weights:
        scores = [sdr_db(target, blend_audio(sources, weights)) for sources, target in loaded]
        mean_sdr = float(np.mean(scores)) if scores else -float("inf")
        if mean_sdr > best_mean_sdr:
            best_mean_sdr = mean_sdr
            best_weights = weights
    if best_weights is None:
        raise ValueError("no candidate weights were evaluated")
    return best_weights, {
        "candidate_weight_count": len(candidate_weights),
        "train_mean_sdr_db": round(best_mean_sdr, 6),
        "weight_grid_step": float(weight_grid_step),
    }


def failed_metric_checks(*, candidate_sdr: float, improvement: float, min_sdr_db: float, min_improvement_db: float) -> list[str]:
    failed: list[str] = []
    if candidate_sdr < float(min_sdr_db):
        failed.append("sdr_below_threshold")
    if improvement < float(min_improvement_db):
        failed.append("improvement_below_threshold")
    return failed


def evaluate_holdout_groups(
    holdout_groups: Sequence[dict[str, Any]],
    *,
    families: Sequence[str],
    weights: Sequence[float],
    output_dir: Path,
    stem: str,
    config_id: str,
    training_track_ids: Sequence[str],
    min_sdr_db: float,
    min_improvement_db: float,
    max_holdouts_per_config: int,
    progress_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_groups = list(holdout_groups[: int(max_holdouts_per_config)]) if int(max_holdouts_per_config) > 0 else list(holdout_groups)
    for holdout_index, group in enumerate(selected_groups, start=1):
        progress(progress_path, "holdout_group_start", config_id=config_id, holdout_index=holdout_index, group_id=group["group_id"])
        sources, target, sample_rate = load_group_audio(group, families)
        family_sdrs = {family: round(sdr_db(target, source), 6) for family, source in zip(families, sources)}
        baseline_family = max(family_sdrs, key=lambda family: family_sdrs[family])
        baseline_sdr = float(family_sdrs[baseline_family])
        candidate = blend_audio(sources, weights)
        candidate_sdr = sdr_db(target, candidate)
        improvement = candidate_sdr - baseline_sdr
        failed_checks = failed_metric_checks(
            candidate_sdr=candidate_sdr,
            improvement=improvement,
            min_sdr_db=min_sdr_db,
            min_improvement_db=min_improvement_db,
        )
        candidate_path = output_dir / "holdout_audio" / f"{config_id}_{group['group_id']}_{stem}.flac"
        write_flac(candidate_path, candidate, sample_rate)
        rows.append(
            {
                "schema": "candidate_family_blend_holdout_row.v1",
                "status": "passed" if not failed_checks else "failed",
                "failed_checks": failed_checks,
                "config_id": config_id,
                "stem": stem,
                "holdout_track_id": group["track_id"],
                "holdout_group_id": group["group_id"],
                "checkpoint_training_track_ids": list(training_track_ids),
                "families": list(families),
                "weights": {family: round(float(weight), 10) for family, weight in zip(families, weights)},
                "baseline_policy": "best_family",
                "baseline_family": baseline_family,
                "family_sdr_db": family_sdrs,
                "metrics": {
                    "baseline_sdr_db": round(baseline_sdr, 6),
                    "candidate_sdr_db": round(candidate_sdr, 6),
                    "improvement_db": round(improvement, 6),
                },
                "candidate_audio_path": str(candidate_path),
                "expected_acceleration_mode": "external_only_cpu_dsp_training_and_eval",
                "gpu_inference_skipped": True,
                "gpu_inference_skip_reason": "candidate_family_blend_is_cpu_dsp_weight_search",
                "delivery_allowed": False,
                "release_ready": False,
            }
        )
        progress(
            progress_path,
            "holdout_group_finish",
            config_id=config_id,
            holdout_index=holdout_index,
            status=rows[-1]["status"],
            failed_checks=failed_checks,
            candidate_sdr_db=round(candidate_sdr, 6),
            improvement_db=round(improvement, 6),
        )
    return rows


def metrics_from_holdout_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidate_sdrs = [float((row.get("metrics") or {}).get("candidate_sdr_db")) for row in rows]
    baseline_sdrs = [float((row.get("metrics") or {}).get("baseline_sdr_db")) for row in rows]
    improvements = [float((row.get("metrics") or {}).get("improvement_db")) for row in rows]
    if not rows:
        return {
            "min_candidate_sdr_db": None,
            "mean_candidate_sdr_db": None,
            "min_baseline_sdr_db": None,
            "mean_baseline_sdr_db": None,
            "min_improvement_db": None,
            "mean_improvement_db": None,
        }
    return {
        "min_candidate_sdr_db": round(min(candidate_sdrs), 6),
        "mean_candidate_sdr_db": round(sum(candidate_sdrs) / len(candidate_sdrs), 6),
        "min_baseline_sdr_db": round(min(baseline_sdrs), 6),
        "mean_baseline_sdr_db": round(sum(baseline_sdrs) / len(baseline_sdrs), 6),
        "min_improvement_db": round(min(improvements), 6),
        "mean_improvement_db": round(sum(improvements) / len(improvements), 6),
    }


def config_list(config_matrix: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    raw_configs = config_matrix.get("configs") if isinstance(config_matrix, dict) else config_matrix
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("config matrix must be a non-empty list or contain a non-empty configs list")
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_configs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"matrix config #{index} must be an object")
        config_id = str(item.get("id") or item.get("config_id") or f"config_{index:03d}").strip()
        if not config_id:
            raise ValueError(f"matrix config #{index} has empty id")
        if config_id in seen:
            raise ValueError(f"duplicate matrix config id: {config_id}")
        seen.add(config_id)
        configs.append({**item, "id": config_id})
    return configs


def _config_families(config: dict[str, Any]) -> list[str]:
    families = config.get("families")
    if not isinstance(families, list) or len(families) < 2:
        raise ValueError(f"config {config['id']}: families must contain at least two entries")
    normalized = [str(family).strip() for family in families if str(family).strip()]
    if len(normalized) != len(families) or len(set(normalized)) != len(normalized):
        raise ValueError(f"config {config['id']}: families must be non-empty and unique")
    return normalized


def _row_contract_failed_checks(holdout_rows: Sequence[dict[str, Any]]) -> list[str]:
    failed: list[str] = []
    if not holdout_rows:
        failed.append("holdout_rows_missing")
    for row in holdout_rows:
        if row.get("status") != "passed":
            failed.append("holdout_row_failed")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        if metrics.get("candidate_sdr_db") is None or metrics.get("improvement_db") is None:
            failed.append("holdout_metrics_missing")
        for check in row.get("failed_checks") or []:
            if str(check).strip() and check not in failed:
                failed.append(str(check))
        if row.get("delivery_allowed") is True:
            failed.append("holdout_delivery_allowed_true")
        if row.get("release_ready") is True:
            failed.append("holdout_release_ready_true")
    if "sdr_below_threshold" in failed or "improvement_below_threshold" in failed:
        failed.append("holdout_metric_below_threshold")
    return sorted(set(failed))


def run_candidate_family_blend_matrix(
    *,
    audio_training_examples_jsonl: Path,
    config_matrix_json: Path,
    output_dir: Path,
    stem: str,
    scope: str,
    exclude_holdout_track_ids: set[str] | None = None,
    min_sdr_db: float = 3.0,
    min_improvement_db: float = 0.5,
    max_holdouts_per_config: int = 2,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    stem = validate_stem_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    exclude_holdout_track_ids = {str(item) for item in (exclude_holdout_track_ids or set()) if str(item)}
    configs = config_list(read_json(config_matrix_json))
    rows = read_jsonl(audio_training_examples_jsonl)
    progress(
        progress_path,
        "start",
        audio_training_examples_jsonl=str(audio_training_examples_jsonl),
        config_matrix_json=str(config_matrix_json),
        config_count=len(configs),
        stem=stem,
        scope=scope,
        excluded_holdout_track_ids=sorted(exclude_holdout_track_ids),
    )

    matrix_rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs, start=1):
        config_id = str(config["id"])
        families = _config_families(config)
        weight_grid_step = float(config.get("weight_grid_step", 0.05))
        progress(progress_path, "config_start", config_id=config_id, config_index=config_index, families=families)
        config_root = output_dir / "configs" / f"{config_index:03d}_{config_id}"
        train_dir = config_root / "train"
        holdout_dir = config_root / "holdout"
        train_dir.mkdir(parents=True, exist_ok=True)
        holdout_dir.mkdir(parents=True, exist_ok=True)
        complete_groups, skipped_groups = complete_candidate_groups(rows, stem=stem, families=families)
        training_groups = [group for group in complete_groups if group["track_id"] not in exclude_holdout_track_ids]
        holdout_groups = (
            [group for group in complete_groups if group["track_id"] in exclude_holdout_track_ids]
            if exclude_holdout_track_ids
            else []
        )
        failed_checks: list[str] = []
        if not training_groups:
            failed_checks.append("training_complete_groups_missing")
        if not holdout_groups:
            failed_checks.append("holdout_complete_groups_missing")

        checkpoint_path = train_dir / CHECKPOINT_NAME
        run_rows_path = train_dir / RUN_ROWS_NAME
        holdout_rows_path = holdout_dir / HOLDOUT_ROWS_NAME
        holdout_rows: list[dict[str, Any]] = []
        weights: list[float] = []
        train_diagnostics: dict[str, Any] = {}
        training_track_ids = sorted({str(group["track_id"]) for group in training_groups})
        if not failed_checks:
            weights, train_diagnostics = train_global_weights(
                training_groups,
                families=families,
                weight_grid_step=weight_grid_step,
                progress_path=progress_path,
                config_id=config_id,
            )
            checkpoint = {
                "schema": "candidate_family_blend_adapter.v1",
                "adapter_id": stable_hash(
                    {
                        "scope": scope,
                        "config_id": config_id,
                        "families": families,
                        "weights": weights,
                        "training_track_ids": training_track_ids,
                    }
                )[:16],
                "config_id": config_id,
                "scope": f"{scope}__{config_id}",
                "stem": stem,
                "families": families,
                "weights": {family: round(float(weight), 10) for family, weight in zip(families, weights)},
                "training_track_ids": training_track_ids,
                "training_group_count": len(training_groups),
                "holdout_track_ids_excluded_from_training": sorted(exclude_holdout_track_ids),
                "baseline_policy": "best_family",
                "release_ready": False,
                "delivery_allowed": False,
                "train_diagnostics": train_diagnostics,
            }
            write_json(checkpoint_path, checkpoint)
            run_rows = [
                {
                    "schema": "candidate_family_blend_run_row.v1",
                    "status": "executed",
                    "scope": f"{scope}__{config_id}",
                    "example_id": checkpoint["adapter_id"],
                    "track_id": checkpoint["adapter_id"],
                    "training_track_ids": training_track_ids,
                    "families": families,
                    "weights": checkpoint["weights"],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "expected_acceleration_mode": "external_only_cpu_dsp_training_and_eval",
                    "gpu_inference_skipped": True,
                    "gpu_training_skipped": True,
                    "delivery_allowed": False,
                    "release_ready": False,
                }
            ]
            write_jsonl(run_rows_path, run_rows)
            holdout_rows = evaluate_holdout_groups(
                holdout_groups,
                families=families,
                weights=weights,
                output_dir=holdout_dir,
                stem=stem,
                config_id=config_id,
                training_track_ids=training_track_ids,
                min_sdr_db=min_sdr_db,
                min_improvement_db=min_improvement_db,
                max_holdouts_per_config=max_holdouts_per_config,
                progress_path=progress_path,
            )
            write_jsonl(holdout_rows_path, holdout_rows)
            failed_checks.extend(_row_contract_failed_checks(holdout_rows))
        else:
            write_jsonl(run_rows_path, [])
            write_jsonl(holdout_rows_path, [])
        failed_checks = sorted(set(failed_checks))
        passed_eval_count = sum(1 for row in holdout_rows if row.get("status") == "passed")
        failed_eval_count = len(holdout_rows) - passed_eval_count
        row = {
            "schema": "generic_audio_finetune_matrix_row.v1",
            "status": "passed" if not failed_checks else "failed",
            "failed_checks": failed_checks,
            "config_id": config_id,
            "scope": f"{scope}__{config_id}",
            "stem": stem,
            "families": families,
            "train_config": {
                "families": families,
                "weight_grid_step": weight_grid_step,
                "baseline_policy": "best_family",
                "max_holdouts_per_config": int(max_holdouts_per_config),
            },
            "train_diagnostics": train_diagnostics,
            "training_group_count": len(training_groups),
            "training_track_ids": training_track_ids,
            "skipped_or_incomplete_group_count": len(skipped_groups),
            "holdout_eval_count": len(holdout_rows),
            "passed_eval_count": passed_eval_count,
            "failed_eval_count": failed_eval_count,
            "metrics": metrics_from_holdout_rows(holdout_rows),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else "",
            "run_rows_jsonl": str(run_rows_path),
            "holdout_rows_jsonl": str(holdout_rows_path),
            "delivery_allowed": False,
            "release_ready": False,
        }
        matrix_rows.append(row)
        progress(
            progress_path,
            "config_finish",
            config_id=config_id,
            config_index=config_index,
            status=row["status"],
            failed_checks=failed_checks,
            holdout_eval_count=len(holdout_rows),
            passed_eval_count=passed_eval_count,
        )

    rows_path = output_dir / MATRIX_ROWS_NAME
    write_jsonl(rows_path, matrix_rows)
    passed_rows = [row for row in matrix_rows if row.get("status") == "passed"]
    best_row = max(
        passed_rows,
        key=lambda row: float((row.get("metrics") or {}).get("mean_improvement_db") or -float("inf")),
        default=None,
    )
    manifest = {
        "schema": "generic_audio_finetune_matrix_manifest.v1",
        "matrix_type": "candidate_family_blend_matrix",
        "scope": scope,
        "stem": stem,
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "config_matrix_json": str(config_matrix_json),
        "output_dir": str(output_dir),
        "matrix_rows_jsonl": str(rows_path),
        "progress_jsonl": str(progress_path),
        "config_count": len(configs),
        "passed_config_count": len(passed_rows),
        "failed_config_count": len(matrix_rows) - len(passed_rows),
        "matrix_validation_status": "passed" if best_row is not None else "failed",
        "best_config_id": str(best_row.get("config_id") if best_row else ""),
        "best_matrix_row": best_row,
        "training_started": True,
        "fine_tune_started": True,
        "models_changed": True,
        "model_registry_edited": False,
        "runtime_gates_changed": False,
        "thresholds_relaxed": False,
        "delivery_allowed": False,
        "promotion_allowed": False,
        "presentation_status": "training_matrix_artifact_not_for_listen_pack",
        "release_status": "blocked_until_holdout_and_promotion_gates",
        "release_ready": False,
        "command_args": command_args or [],
        "contract": {
            "candidate_family_blend_is_trainable_adapter": True,
            "baseline_policy": "best_family",
            "holdout_uses_verified_examples_only": True,
            "holdout_tracks_excluded_from_training": True,
            "pseudo_teacher_rows_used": False,
            "release_ready_forced_false": True,
            "fallback_used": False,
        },
    }
    write_json(output_dir / MATRIX_MANIFEST_NAME, manifest)
    progress(
        progress_path,
        "finish",
        matrix_validation_status=manifest["matrix_validation_status"],
        best_config_id=manifest["best_config_id"],
        passed_config_count=manifest["passed_config_count"],
        failed_config_count=manifest["failed_config_count"],
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict candidate-family blend adapter matrix.")
    parser.add_argument("--audio-training-examples-jsonl", type=Path, required=True)
    parser.add_argument("--config-matrix-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vocal")
    parser.add_argument("--scope", default="candidate_family_blend_matrix")
    parser.add_argument("--exclude-holdout-track-id", action="append", default=[])
    parser.add_argument("--min-sdr-db", type=float, default=3.0)
    parser.add_argument("--min-improvement-db", type=float, default=0.5)
    parser.add_argument("--max-holdouts-per-config", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_candidate_family_blend_matrix.py",
        entrypoint_path="/app/tools/run_candidate_family_blend_matrix.py",
    )
    args = parse_args(argv)
    command_args = list(argv if argv is not None else sys.argv[1:])
    manifest = run_candidate_family_blend_matrix(
        audio_training_examples_jsonl=args.audio_training_examples_jsonl,
        config_matrix_json=args.config_matrix_json,
        output_dir=args.output_dir,
        stem=str(args.stem),
        scope=str(args.scope),
        exclude_holdout_track_ids={str(item) for item in args.exclude_holdout_track_id},
        min_sdr_db=float(args.min_sdr_db),
        min_improvement_db=float(args.min_improvement_db),
        max_holdouts_per_config=int(args.max_holdouts_per_config),
        command_args=command_args,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MATRIX_MANIFEST_NAME),
                "matrix_validation_status": manifest["matrix_validation_status"],
                "best_config_id": manifest["best_config_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["matrix_validation_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
