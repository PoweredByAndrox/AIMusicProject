#!/usr/bin/env python3
"""Bounded benchmark for Mac MelBand refiner checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.apply_mac_stem_refiner_to_mel_guard import run_refiner_replay
from tools.mac_melband_pitch_guard import (
    pitch_shift_resample_same_length,
    score_best_value,
    score_pitch_shifted_vocal_leak,
    score_region_count,
    score_target_issue_improved,
)
from tools.stem_region_repair_common import read_audio_stereo, write_audio, write_json
from tools.stem_region_repair_model import resolve_region_repair_training_device


ALLOWED_BENCHMARK_OUTCOMES = {
    "promote_to_next_diagnostic_stage",
    "reject_as_overfit",
    "model_gap_requires_new_training_objective",
}
LEAK_ROW_TYPES = {"source_hardcase", "region_hardcase", "synthetic_variant"}
DEFAULT_CHECKPOINT_IDS = ("g1.5", "g2.0", "g2.5")
REGION_HARDCASE_PAD_SEC = 1.0


def _append_progress(path: Path, phase: str, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_audio(*arrays: np.ndarray) -> list[np.ndarray]:
    frames = min(array.shape[0] for array in arrays)
    return [np.asarray(array[:frames], dtype=np.float32) for array in arrays]


def _resolve_audio(path_value: str, *, guard_dir: Path) -> Path:
    path = Path(str(path_value or ""))
    if path.exists():
        return path
    candidate = guard_dir / "audio" / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"benchmark audio path not found: {path_value}")


def _safe_checkpoint_alias(value: str) -> str:
    alias = str(value).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not alias or len(alias) > 16 or any(char not in allowed for char in alias):
        raise ValueError(f"unsafe checkpoint id: {value!r}")
    return alias


def validate_benchmark_row(row: Mapping[str, Any]) -> None:
    row_id = str(row.get("row_id") or "")
    if not row_id or len(row_id) > 32:
        raise ValueError(f"invalid benchmark row_id: {row_id!r}")
    label_type = str(row.get("label_type") or "")
    if label_type == "pseudo_teacher" and bool(row.get("is_ground_truth")):
        raise ValueError("pseudo_teacher benchmark rows cannot be ground truth")
    if bool(row.get("delivery_allowed")):
        raise ValueError("benchmark rows cannot set delivery_allowed=true")
    if str(row.get("row_type") or "") not in LEAK_ROW_TYPES | {"clean_negative"}:
        raise ValueError(f"unsupported benchmark row_type: {row.get('row_type')!r}")


def _valid_for_decision(row: Mapping[str, Any]) -> bool:
    row_type = str(row.get("row_type") or "")
    if row_type in LEAK_ROW_TYPES and int(row.get("before_regions") or 0) <= 0:
        return False
    return bool(row.get("valid_for_decision", True))


def _row_passes(row: Mapping[str, Any]) -> bool:
    validate_benchmark_row(row)
    if not _valid_for_decision(row):
        return False
    row_type = str(row["row_type"])
    if bool(row.get("damage_regressed")):
        return False
    if row_type == "clean_negative":
        return int(row.get("after_regions") or 0) <= int(row.get("before_regions") or 0)
    return bool(row.get("target_issue_improved")) and int(row.get("after_regions") or 0) == 0


def _count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(count, (int, float)):
            result[str(key)] = max(0, int(count))
    return dict(sorted(result.items()))


def _score_failure_taxonomy(score: Mapping[str, Any] | None) -> dict[str, Any]:
    taxonomy = score.get("failure_taxonomy") if isinstance(score, Mapping) else None
    taxonomy_map = taxonomy if isinstance(taxonomy, Mapping) else {}
    if not taxonomy_map and isinstance(score, Mapping) and ("by_fail_mode" in score or "by_probe" in score):
        taxonomy_map = score
    total_fail_regions = score_region_count(score)
    selected_count = taxonomy_map.get("selected_region_count")
    if not isinstance(selected_count, (int, float)):
        selected_count = len(score.get("regions") or []) if isinstance(score, Mapping) else 0
    return {
        "total_fail_regions": int(total_fail_regions),
        "selected_region_count": max(0, int(selected_count)),
        "regions_capped": bool(taxonomy_map.get("regions_capped") or (int(total_fail_regions) > int(selected_count))),
        "by_fail_mode": _count_map(taxonomy_map.get("by_fail_mode")),
        "by_probe": _count_map(taxonomy_map.get("by_probe")),
    }


def _merge_count_maps(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[str(key)] = target.get(str(key), 0) + max(0, int(value))


def _aggregate_taxonomy(rows: list[dict[str, Any]], taxonomy_key: str, region_key: str) -> dict[str, Any]:
    by_fail_mode: dict[str, int] = {}
    by_probe: dict[str, int] = {}
    total = 0
    selected = 0
    capped_rows = 0
    for row in rows:
        taxonomy = _score_failure_taxonomy(row.get(taxonomy_key) if isinstance(row.get(taxonomy_key), Mapping) else None)
        if taxonomy["total_fail_regions"] == 0 and taxonomy_key not in row:
            taxonomy["total_fail_regions"] = max(0, int(row.get(region_key) or 0))
        total += int(taxonomy["total_fail_regions"])
        selected += int(taxonomy["selected_region_count"])
        capped_rows += 1 if bool(taxonomy.get("regions_capped")) else 0
        _merge_count_maps(by_fail_mode, taxonomy.get("by_fail_mode") or {})
        _merge_count_maps(by_probe, taxonomy.get("by_probe") or {})
    return {
        "total_fail_regions": int(total),
        "selected_region_count": int(selected),
        "capped_row_count": int(capped_rows),
        "by_fail_mode": dict(sorted(by_fail_mode.items())),
        "by_probe": dict(sorted(by_probe.items())),
    }


def _new_group_summary() -> dict[str, Any]:
    return {
        "row_count": 0,
        "before_regions": 0,
        "after_regions": 0,
        "region_delta": 0,
        "failing_rows_after": 0,
        "damage_regression_rows": 0,
    }


def _add_row_to_group(group: dict[str, Any], row: Mapping[str, Any]) -> None:
    before_regions = max(0, int(row.get("before_regions") or 0))
    after_regions = max(0, int(row.get("after_regions") or 0))
    group["row_count"] += 1
    group["before_regions"] += before_regions
    group["after_regions"] += after_regions
    group["region_delta"] += before_regions - after_regions
    group["failing_rows_after"] += 1 if after_regions > 0 else 0
    group["damage_regression_rows"] += 1 if bool(row.get("damage_regressed")) else 0


def _unresolved_row(row: Mapping[str, Any]) -> dict[str, Any]:
    after_taxonomy = _score_failure_taxonomy(row.get("after_failure_taxonomy") if isinstance(row.get("after_failure_taxonomy"), Mapping) else None)
    before_regions = max(0, int(row.get("before_regions") or 0))
    after_regions = max(0, int(row.get("after_regions") or 0))
    return {
        "row_id": str(row.get("row_id") or ""),
        "row_type": str(row.get("row_type") or ""),
        "checkpoint_id": str(row.get("checkpoint_id") or ""),
        "before_regions": before_regions,
        "after_regions": after_regions,
        "region_delta": before_regions - after_regions,
        "after_best_score": float(row.get("after_best_score") or 0.0),
        "target_issue_improved": bool(row.get("target_issue_improved")),
        "damage_regressed": bool(row.get("damage_regressed")),
        "after_by_fail_mode": after_taxonomy["by_fail_mode"],
        "after_by_probe": after_taxonomy["by_probe"],
    }


def summarize_benchmark_taxonomy(rows: Iterable[Mapping[str, Any]], *, checkpoint_order: list[str]) -> dict[str, Any]:
    row_list = [dict(row) for row in rows]
    for row in row_list:
        validate_benchmark_row(row)
    total_before = sum(max(0, int(row.get("before_regions") or 0)) for row in row_list)
    total_after = sum(max(0, int(row.get("after_regions") or 0)) for row in row_list)
    by_checkpoint: dict[str, dict[str, Any]] = {}
    by_row_type: dict[str, dict[str, Any]] = {}
    for row in row_list:
        checkpoint_id = str(row.get("checkpoint_id") or "")
        row_type = str(row.get("row_type") or "")
        by_checkpoint.setdefault(checkpoint_id, _new_group_summary())
        by_row_type.setdefault(row_type, _new_group_summary())
        _add_row_to_group(by_checkpoint[checkpoint_id], row)
        _add_row_to_group(by_row_type[row_type], row)

    checkpoint_ids = list(dict.fromkeys([*checkpoint_order, *sorted(by_checkpoint)]))
    checkpoint_rank = [
        {"checkpoint_id": checkpoint_id, **by_checkpoint[checkpoint_id]}
        for checkpoint_id in checkpoint_ids
        if checkpoint_id in by_checkpoint
    ]
    checkpoint_rank.sort(
        key=lambda item: (
            int(item["after_regions"]),
            int(item["damage_regression_rows"]),
            int(item["failing_rows_after"]),
            -int(item["region_delta"]),
            str(item["checkpoint_id"]),
        )
    )
    unresolved_rows = [
        _unresolved_row(row)
        for row in row_list
        if not _row_passes(row)
    ]
    unresolved_rows.sort(
        key=lambda item: (
            -int(item["after_regions"]),
            -float(item["after_best_score"]),
            str(item["checkpoint_id"]),
            str(item["row_id"]),
        )
    )
    return {
        "schema": "mac_mel_refiner_benchmark_taxonomy_summary.v1",
        "delivery_allowed": False,
        "presentation_status": "benchmark_taxonomy_diagnostic_only",
        "row_count": len(row_list),
        "total_before_regions": int(total_before),
        "total_after_regions": int(total_after),
        "total_region_delta": int(total_before - total_after),
        "before_failure_taxonomy": _aggregate_taxonomy(row_list, "before_failure_taxonomy", "before_regions"),
        "after_failure_taxonomy": _aggregate_taxonomy(row_list, "after_failure_taxonomy", "after_regions"),
        "by_checkpoint": {key: by_checkpoint[key] for key in sorted(by_checkpoint)},
        "by_row_type": {key: by_row_type[key] for key in sorted(by_row_type)},
        "checkpoint_rank": checkpoint_rank,
        "unresolved_rows": unresolved_rows[:20],
    }


def decide_benchmark_outcome(rows: Iterable[Mapping[str, Any]], *, checkpoint_order: list[str]) -> dict[str, Any]:
    row_list = [dict(row) for row in rows]
    for row in row_list:
        validate_benchmark_row(row)
    invalid_row_count = sum(1 for row in row_list if not _valid_for_decision(row))
    decision_rows = [row for row in row_list if _valid_for_decision(row)]
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in decision_rows:
        by_checkpoint.setdefault(str(row["checkpoint_id"]), []).append(row)

    source_only_success = False
    any_clean_regression = False
    for checkpoint_id in checkpoint_order:
        checkpoint_rows = by_checkpoint.get(str(checkpoint_id), [])
        if not checkpoint_rows:
            continue
        clean_rows = [row for row in checkpoint_rows if row["row_type"] == "clean_negative"]
        leak_rows = [row for row in checkpoint_rows if row["row_type"] in LEAK_ROW_TYPES]
        source_rows = [row for row in checkpoint_rows if row["row_type"] == "source_hardcase"]
        non_source_leak_rows = [row for row in leak_rows if row["row_type"] != "source_hardcase"]
        clean_pass = all(_row_passes(row) for row in clean_rows)
        leak_pass = all(_row_passes(row) for row in leak_rows)
        if any(not _row_passes(row) for row in clean_rows):
            any_clean_regression = True
        if source_rows and all(_row_passes(row) for row in source_rows) and any(not _row_passes(row) for row in non_source_leak_rows):
            source_only_success = True
        if clean_rows and leak_rows and clean_pass and leak_pass:
            if invalid_row_count > 0:
                return {
                    "outcome": "reject_as_overfit",
                    "selected_checkpoint_id": None,
                    "delivery_allowed": False,
                    "checkpoint_count": len(checkpoint_order),
                    "row_count": len(decision_rows),
                    "invalid_row_count": invalid_row_count,
                    "blocked_reason": "invalid_leak_rows_without_before_defect",
                }
            return {
                "outcome": "promote_to_next_diagnostic_stage",
                "selected_checkpoint_id": str(checkpoint_id),
                "delivery_allowed": False,
                "checkpoint_count": len(checkpoint_order),
                "row_count": len(decision_rows),
                "invalid_row_count": invalid_row_count,
            }
    outcome = "model_gap_requires_new_training_objective" if any_clean_regression or not source_only_success else "reject_as_overfit"
    return {
        "outcome": outcome,
        "selected_checkpoint_id": None,
        "delivery_allowed": False,
        "checkpoint_count": len(checkpoint_order),
        "row_count": len(decision_rows),
        "invalid_row_count": invalid_row_count,
    }


def _write_guard_manifest(
    *,
    guard_dir: Path,
    vocals_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    raw_audio: np.ndarray,
    sample_rate: int,
    config: Mapping[str, Any],
) -> Path:
    audio_dir = guard_dir / "audio"
    vocals_path = write_audio(audio_dir / "v.flac", vocals_audio, sample_rate)
    instrumental_path = write_audio(audio_dir / "i.flac", instrumental_audio, sample_rate)
    raw_path = write_audio(audio_dir / "raw.flac", raw_audio, sample_rate)
    manifest = {
        "schema": "mac_melband_benchmark_guard.v1",
        "audio": {
            "vocals": str(vocals_path),
            "instrumental": str(instrumental_path),
            "raw_ref": str(raw_path),
        },
        "score": {"config": dict(config)},
        "delivery_allowed": False,
        "presentation_status": "benchmark_guard_only",
    }
    return write_json(guard_dir / "manifest.json", manifest)


def _crop(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, min(audio.shape[0], int(round(float(start_sec) * sample_rate))))
    end = max(start, min(audio.shape[0], int(round(float(end_sec) * sample_rate))))
    return np.asarray(audio[start:end], dtype=np.float32)


def build_benchmark_guard_rows(*, mel_guard_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    guard_manifest = _load_json(Path(mel_guard_dir) / "manifest.json")
    audio_map = dict(guard_manifest.get("audio") or {})
    config = dict((guard_manifest.get("score") or {}).get("config") or {})
    vocals_path = _resolve_audio(str(audio_map.get("vocals")), guard_dir=Path(mel_guard_dir))
    instrumental_path = _resolve_audio(str(audio_map.get("instrumental")), guard_dir=Path(mel_guard_dir))
    raw_path = _resolve_audio(str(audio_map.get("raw_ref")), guard_dir=Path(mel_guard_dir))
    vocals_audio, vocals_rate = read_audio_stereo(vocals_path)
    instrumental_audio, instrumental_rate = read_audio_stereo(instrumental_path)
    raw_audio, raw_rate = read_audio_stereo(raw_path)
    if len({vocals_rate, instrumental_rate, raw_rate}) != 1:
        raise ValueError("benchmark_source_sample_rate_mismatch")
    sample_rate = vocals_rate
    vocals_audio, instrumental_audio, raw_audio = _fit_audio(vocals_audio, instrumental_audio, raw_audio)

    rows: list[dict[str, Any]] = []
    source_guard_dir = output_dir / "guards" / "src"
    shutil.copytree(mel_guard_dir, source_guard_dir, dirs_exist_ok=True)
    rows.append({"row_id": "src", "row_type": "source_hardcase", "guard_dir": str(source_guard_dir)})

    regions = list((guard_manifest.get("score") or {}).get("regions") or [])[:2]
    for index, region in enumerate(regions, start=1):
        start_sec = max(0.0, float(region.get("start", 0.0)) - REGION_HARDCASE_PAD_SEC)
        end_sec = min(raw_audio.shape[0] / float(sample_rate), float(region.get("end", 0.0)) + REGION_HARDCASE_PAD_SEC)
        if end_sec <= start_sec:
            continue
        row_id = f"reg{index:02d}"
        guard_dir = output_dir / "guards" / row_id
        v = _crop(vocals_audio, sample_rate, start_sec, end_sec)
        i = _crop(instrumental_audio, sample_rate, start_sec, end_sec)
        raw = _crop(raw_audio, sample_rate, start_sec, end_sec)
        _write_guard_manifest(guard_dir=guard_dir, vocals_audio=v, instrumental_audio=i, raw_audio=raw, sample_rate=sample_rate, config=config)
        rows.append({"row_id": row_id, "row_type": "region_hardcase", "guard_dir": str(guard_dir)})

    synth_frames = min(sample_rate * 2, vocals_audio.shape[0])
    synth_vocals = vocals_audio[:synth_frames]
    t = np.arange(synth_frames, dtype=np.float32) / float(sample_rate)
    clean_instrumental = np.stack(
        [
            0.055 * np.sin(2.0 * math.pi * 146.83 * t) + 0.035 * np.sin(2.0 * math.pi * 440.0 * t),
            0.052 * np.sin(2.0 * math.pi * 150.0 * t) + 0.033 * np.sin(2.0 * math.pi * 445.0 * t),
        ],
        axis=1,
    ).astype(np.float32)
    leak = pitch_shift_resample_same_length(synth_vocals, -5.0) * 0.34
    synth_guard_dir = output_dir / "guards" / "syn01"
    synth_instrumental = np.clip(clean_instrumental + leak, -1.0, 1.0).astype(np.float32)
    _write_guard_manifest(
        guard_dir=synth_guard_dir,
        vocals_audio=synth_vocals,
        instrumental_audio=synth_instrumental,
        raw_audio=np.clip(synth_vocals + synth_instrumental, -1.0, 1.0).astype(np.float32),
        sample_rate=sample_rate,
        config=config,
    )
    rows.append({"row_id": "syn01", "row_type": "synthetic_variant", "guard_dir": str(synth_guard_dir)})

    clean_guard_dir = output_dir / "guards" / "cln01"
    _write_guard_manifest(
        guard_dir=clean_guard_dir,
        vocals_audio=synth_vocals,
        instrumental_audio=clean_instrumental,
        raw_audio=np.clip(synth_vocals + clean_instrumental, -1.0, 1.0).astype(np.float32),
        sample_rate=sample_rate,
        config=config,
    )
    rows.append({"row_id": "cln01", "row_type": "clean_negative", "guard_dir": str(clean_guard_dir)})
    return rows


def _score_clean_row(*, row: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    guard_dir = Path(str(row["guard_dir"]))
    manifest = _load_json(guard_dir / "manifest.json")
    audio_map = dict(manifest.get("audio") or {})
    before = score_pitch_shifted_vocal_leak(
        vocals_path=Path(str(audio_map["vocals"])),
        instrumental_path=Path(str(audio_map["instrumental"])),
        output_dir=output_dir / "score" / "before",
        config=(manifest.get("score") or {}).get("config"),
    )
    after = score_pitch_shifted_vocal_leak(
        vocals_path=Path(str(audio_map["vocals"])),
        instrumental_path=Path(str(audio_map["instrumental"])),
        output_dir=output_dir / "score" / "after",
        config=before.get("config"),
    )
    before_regions = score_region_count(before)
    after_regions = score_region_count(after)
    return {
        "before_score": before,
        "after_score": after,
        "before_regions": before_regions,
        "after_regions": after_regions,
        "before_failure_taxonomy": _score_failure_taxonomy(before),
        "after_failure_taxonomy": _score_failure_taxonomy(after),
        "before_best_score": score_best_value(before),
        "after_best_score": score_best_value(after),
        "target_issue_improved": False,
        "damage_regressed": after_regions > before_regions,
    }


def _benchmark_row_from_replay(
    *,
    row: Mapping[str, Any],
    checkpoint_id: str,
    replay_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = dict(replay_manifest.get("metrics") or {})
    before_score = replay_manifest.get("before_score") if isinstance(replay_manifest.get("before_score"), Mapping) else None
    after_score = replay_manifest.get("after_score") if isinstance(replay_manifest.get("after_score"), Mapping) else None
    before_regions = (
        score_region_count(before_score)
        if before_score is not None
        else int(metrics.get("before_region_count") or 0)
    )
    after_regions = (
        score_region_count(after_score)
        if after_score is not None
        else int(metrics.get("after_region_count") or 0)
    )
    before_best_score = score_best_value(before_score) if before_score is not None else float(metrics.get("before_best_score") or 0.0)
    after_best_score = score_best_value(after_score) if after_score is not None else float(metrics.get("after_best_score") or 0.0)
    if before_score is not None and after_score is not None:
        target_issue_improved = score_target_issue_improved(before_score=before_score, after_score=after_score)
    else:
        target_issue_improved = bool(metrics.get("target_issue_improved"))
    return {
        "row_id": str(row["row_id"]),
        "row_type": str(row["row_type"]),
        "checkpoint_id": str(checkpoint_id),
        "before_regions": before_regions,
        "after_regions": after_regions,
        "before_failure_taxonomy": _score_failure_taxonomy(before_score),
        "after_failure_taxonomy": _score_failure_taxonomy(after_score),
        "before_best_score": before_best_score,
        "after_best_score": after_best_score,
        "target_issue_improved": bool(target_issue_improved),
        "damage_regressed": False,
        "delivery_allowed": False,
        "label_type": "pseudo_teacher",
        "is_ground_truth": False,
        "valid_for_decision": before_regions > 0,
        "replay_manifest": str(replay_manifest.get("manifest_path") or ""),
    }


def run_benchmark(
    *,
    mel_guard_dir: Path,
    checkpoint_root: Path,
    output_dir: Path,
    checkpoint_ids: list[str] | None = None,
    require_mps: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    checkpoint_ids = [_safe_checkpoint_alias(value) for value in (checkpoint_ids or list(DEFAULT_CHECKPOINT_IDS))]
    _append_progress(progress_path, "preflight", "start", checkpoint_ids=checkpoint_ids, require_mps=bool(require_mps))
    device = resolve_region_repair_training_device("mps")
    if bool(require_mps) and device.type != "mps":
        manifest = {
            "schema": "mac_mel_refiner_benchmark.v1",
            "outcome": "model_gap_requires_new_training_objective",
            "infra_blocker": "mps_unavailable",
            "delivery_allowed": False,
            "presentation_status": "benchmark_diagnostic_only",
            "resolved_device": str(device),
        }
        write_json(output_dir / "manifest.json", manifest)
        _append_progress(progress_path, "preflight", "blocked_mps_unavailable", resolved_device=str(device))
        return manifest

    guard_rows = build_benchmark_guard_rows(mel_guard_dir=mel_guard_dir, output_dir=output_dir)
    benchmark_rows: list[dict[str, Any]] = []
    before_after_rows: list[dict[str, Any]] = []
    _append_progress(progress_path, "benchmark", "start", row_count=len(guard_rows), checkpoint_count=len(checkpoint_ids))

    for checkpoint_id in checkpoint_ids:
        checkpoint_path = Path(checkpoint_root) / checkpoint_id / "model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        for row in guard_rows:
            row_id = str(row["row_id"])
            row_type = str(row["row_type"])
            row_out = output_dir / "runs" / checkpoint_id / row_id
            _append_progress(progress_path, "row", "start", checkpoint_id=checkpoint_id, row_id=row_id, row_type=row_type)
            if row_type == "clean_negative":
                clean_result = _score_clean_row(row=row, output_dir=row_out)
                bench_row = {
                    "row_id": row_id,
                    "row_type": row_type,
                    "checkpoint_id": checkpoint_id,
                    "before_regions": clean_result["before_regions"],
                    "after_regions": clean_result["after_regions"],
                    "before_failure_taxonomy": clean_result["before_failure_taxonomy"],
                    "after_failure_taxonomy": clean_result["after_failure_taxonomy"],
                    "before_best_score": clean_result["before_best_score"],
                    "after_best_score": clean_result["after_best_score"],
                    "target_issue_improved": clean_result["target_issue_improved"],
                    "damage_regressed": clean_result["damage_regressed"],
                    "delivery_allowed": False,
                    "label_type": "pseudo_teacher",
                    "is_ground_truth": False,
                }
                row_out.mkdir(parents=True, exist_ok=True)
                write_json(row_out / "manifest.json", {"schema": "mac_mel_refiner_clean_row.v1", **bench_row, "delivery_allowed": False})
            else:
                replay_manifest = run_refiner_replay(
                    mel_guard_dir=Path(str(row["guard_dir"])),
                    checkpoint_path=checkpoint_path,
                    output_dir=row_out,
                    require_mps=bool(require_mps),
                    max_regions=8,
                )
                replay_manifest["manifest_path"] = str(row_out / "manifest.json")
                bench_row = _benchmark_row_from_replay(row=row, checkpoint_id=checkpoint_id, replay_manifest=replay_manifest)
            validate_benchmark_row(bench_row)
            benchmark_rows.append(bench_row)
            before_after_rows.append({key: bench_row[key] for key in (
                "row_id",
                "row_type",
                "checkpoint_id",
                "before_regions",
                "after_regions",
                "before_best_score",
                "after_best_score",
                "target_issue_improved",
                "damage_regressed",
                "delivery_allowed",
                "before_failure_taxonomy",
                "after_failure_taxonomy",
            )})
            _append_progress(progress_path, "row", "end", checkpoint_id=checkpoint_id, row_id=row_id, after_regions=bench_row["after_regions"])

    outcome = decide_benchmark_outcome(benchmark_rows, checkpoint_order=checkpoint_ids)
    failure_taxonomy_summary = summarize_benchmark_taxonomy(benchmark_rows, checkpoint_order=checkpoint_ids)
    rows_path = output_dir / "rows.jsonl"
    before_after_path = output_dir / "before_after.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in benchmark_rows) + "\n", encoding="utf-8")
    before_after_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in before_after_rows) + "\n", encoding="utf-8")
    manifest = {
        "schema": "mac_mel_refiner_benchmark.v1",
        "source_guard_dir": str(mel_guard_dir),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_ids": checkpoint_ids,
        "resolved_device": str(device),
        "outcome": outcome["outcome"],
        "selected_checkpoint_id": outcome["selected_checkpoint_id"],
        "blocked_reason": outcome.get("blocked_reason"),
        "rows_jsonl": str(rows_path),
        "before_after_jsonl": str(before_after_path),
        "row_count": len(benchmark_rows),
        "decision_row_count": outcome["row_count"],
        "invalid_row_count": outcome["invalid_row_count"],
        "guard_row_count": len(guard_rows),
        "failure_taxonomy_summary": failure_taxonomy_summary,
        "delivery_allowed": False,
        "listen_pack_created": False,
        "presentation_status": "benchmark_diagnostic_only",
        "contract": {
            "bounded_checkpoint_count": len(checkpoint_ids),
            "allowed_outcomes": sorted(ALLOWED_BENCHMARK_OUTCOMES),
            "no_threshold_relaxation": True,
            "no_listen_delivery": True,
            "pseudo_teacher_rows_ground_truth_allowed": False,
            "taxonomy_summary_diagnostic_only": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    _append_progress(progress_path, "manifest", "end", outcome=manifest["outcome"], selected_checkpoint_id=manifest["selected_checkpoint_id"])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-guard-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-id", action="append", dest="checkpoint_ids")
    parser.add_argument("--require-mps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_benchmark(
        mel_guard_dir=args.mel_guard_dir,
        checkpoint_root=args.checkpoint_root,
        output_dir=args.out,
        checkpoint_ids=list(args.checkpoint_ids or DEFAULT_CHECKPOINT_IDS),
        require_mps=bool(args.require_mps),
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "outcome": manifest["outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
