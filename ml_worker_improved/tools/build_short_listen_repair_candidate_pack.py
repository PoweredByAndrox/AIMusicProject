#!/usr/bin/env python3
"""Build a short-name listen pack of repair candidates from rejected feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import ARTIFACT_AUDIO_DIR, indexed_audio_filename, validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
MINUS_JUDGE_NAME = "minus_judge_candidates.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
READY_QUEUE_STATUSES = {"ready_for_repair_cycle", "ready_for_machine_warning_cycle"}
FULL_PACK_QUEUE_STATUS = "ready_for_full_pack_loop"
FULL_PACK_QUEUE_SOURCE = "machine_full_pack_gate"

BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "deliverable",
    "thresholds_relaxed",
    "runtime_gates_changed",
    "model_registry_edited",
    "models_changed",
    "training_started",
    "fine_tune_started",
    "separator_training_started",
    "cleanup_routes_added",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_repair_candidate_pack",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "repair_candidate_pack_copies_existing_candidates_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "repair_candidate_pack_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(duration, 6),
        "frames": int(info.frames),
    }


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def queue_status_is_ready(queue_status: str, queue_source: str) -> bool:
    if queue_status in READY_QUEUE_STATUSES:
        return True
    return queue_status == FULL_PACK_QUEUE_STATUS and queue_source == FULL_PACK_QUEUE_SOURCE


def collect_true_flag_failures(prefix: str, payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            failed.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                failed.append(f"{prefix}_safety_{key}_true")
    return failed


def canonical_path(value: Any) -> Path:
    return Path(str(value or ""))


def reference_metrics(candidate_path: Path, target_path: Path) -> dict[str, Any]:
    candidate, candidate_sr = sf.read(str(candidate_path), always_2d=True, dtype="float32")
    target, target_sr = sf.read(str(target_path), always_2d=True, dtype="float32")
    if int(candidate_sr) != int(target_sr):
        return {"reference_metric_status": "blocked_sample_rate_mismatch"}
    frames = min(candidate.shape[0], target.shape[0])
    channels = min(candidate.shape[1], target.shape[1])
    if frames <= 0 or channels <= 0:
        return {"reference_metric_status": "blocked_empty_audio"}
    candidate = candidate[:frames, :channels]
    target = target[:frames, :channels]
    residual = target - candidate
    signal = float(np.mean(np.square(target))) + 1.0e-12
    noise = float(np.mean(np.square(residual))) + 1.0e-12
    sdr = 10.0 * math.log10(signal / noise)
    mae = float(np.mean(np.abs(residual)))
    return {
        "reference_metric_status": "computed",
        "reference_sdr_db": round(float(sdr), 6),
        "reference_mae": round(mae, 8),
        "frames_compared": int(frames),
        "sample_rate": int(candidate_sr),
    }


def source_track_id(queue_item: dict[str, Any]) -> str:
    source_row = queue_item.get("source_row") if isinstance(queue_item.get("source_row"), dict) else {}
    return str(source_row.get("track_id") or source_row.get("source_track_id") or queue_item.get("track_id") or "")


def minus_source_manifest_path(queue_item: dict[str, Any]) -> Path:
    def iter_source_row_chain(row: Any) -> Iterable[dict[str, Any]]:
        current = row
        seen: set[int] = set()
        while isinstance(current, dict) and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = current.get("source_row")

    diagnostic_context = queue_item.get("diagnostic_context") if isinstance(queue_item.get("diagnostic_context"), dict) else {}
    minus_audit = diagnostic_context.get("minus_audit") if isinstance(diagnostic_context.get("minus_audit"), dict) else {}
    for source in (
        *iter_source_row_chain(queue_item.get("source_row")),
        *iter_source_row_chain(minus_audit.get("source_row")),
    ):
        nested_path = canonical_path(source.get("source_manifest"))
        if str(nested_path) and nested_path.exists() and nested_path.is_file():
            return nested_path
    return canonical_path(queue_item.get("source_manifest"))


def build_vocal_candidates(
    *,
    queue_item: dict[str, Any],
    examples_path: Path,
    output_dir: Path,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    failed: list[str] = []
    track_id = source_track_id(queue_item)
    if not track_id:
        return [], ["vocal_repair_track_id_missing"]
    if not examples_path.exists():
        return [], ["vocal_training_examples_missing"]

    candidates: list[dict[str, Any]] = []
    seen_source_paths: set[str] = set()
    seen_source_sha256: set[str] = set()
    for example in iter_jsonl(examples_path):
        if str(example.get("track_id") or "") != track_id:
            continue
        if str(example.get("stem_type") or "") != "vocal":
            continue
        if example.get("release_ready") is True:
            failed.append(f"vocal_example_release_ready_true:{example.get('id')}")
            continue
        if str(example.get("candidate_source_status") or "completed") != "completed":
            continue
        source_path = canonical_path(example.get("source_audio_path"))
        target_path = canonical_path(example.get("target_audio_path"))
        if not source_path.exists() or not source_path.is_file():
            failed.append(f"vocal_candidate_audio_missing:{example.get('id')}")
            continue
        source_key = str(source_path)
        if source_key in seen_source_paths:
            continue
        seen_source_paths.add(source_key)
        source_sha256 = sha256_file(source_path)
        if source_sha256 in seen_source_sha256:
            continue
        seen_source_sha256.add(source_sha256)
        metrics = reference_metrics(source_path, target_path) if target_path.exists() else {"reference_metric_status": "target_missing"}
        candidates.append({"example": example, "source_path": source_path, "source_sha256": source_sha256, "reference_metrics": metrics})

    def sort_key(candidate: dict[str, Any]) -> tuple[float, str]:
        metrics = candidate["reference_metrics"]
        sdr = metrics.get("reference_sdr_db")
        return (float(sdr) if isinstance(sdr, (int, float)) else -1.0e9, str(candidate["source_path"]))

    selected = sorted(candidates, key=sort_key, reverse=True)[: max(0, max_candidates)]
    rows: list[dict[str, Any]] = []
    queue_item_id = str(queue_item.get("queue_item_id") or "q000")
    validate_artifact_segment(queue_item_id)
    for index, candidate in enumerate(selected, start=1):
        filename = indexed_audio_filename("v", index)
        listen_path = output_dir / ARTIFACT_AUDIO_DIR / queue_item_id / filename
        listen_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate["source_path"], listen_path)
        metadata = audio_metadata(listen_path)
        example = candidate["example"]
        rows.append(
            {
                "schema": "short_listen_repair_candidate.v1",
                "queue_item_id": queue_item_id,
                "route_intent": "vocal_candidate_replacement",
                "stem_type": "vocal",
                "artifact_alias": filename.removesuffix(".flac"),
                "artifact_filename": filename,
                "artifact_relative_path": str(listen_path.relative_to(output_dir).as_posix()),
                "listen_audio_path": str(listen_path),
                "listen_audio_sha256": sha256_file(listen_path),
                "sample_rate": metadata["sample_rate"],
                "channels": metadata["channels"],
                "duration_sec": metadata["duration_sec"],
                "source_audio_path": str(candidate["source_path"]),
                "source_audio_sha256": str(candidate["source_sha256"]),
                "source_candidate_id": str(example.get("id") or ""),
                "source_candidate_family": str(example.get("source_candidate_family") or ""),
                "source_candidate_model_id": str(example.get("source_candidate_model_id") or ""),
                "source_track_id": track_id,
                "reference_verified": bool(example.get("reference_verified")),
                "target_audio_path": str(example.get("target_audio_path") or ""),
                "reference_metrics": candidate["reference_metrics"],
                "quality_status": "repair_candidate_reference_scored",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_repair_listen_pack",
                "release_status": "listen_preview_only",
                "release_ready": False,
                "can_publish_as_result": False,
                "semantic_quality_verified": False,
            }
        )
    if not rows:
        failed.append(f"no_vocal_repair_candidates:{queue_item_id}")
    return rows, failed


def candidate_id_from_path(path: Path) -> str:
    return path.stem


def phase_variant_candidate_sources(render_manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    phase_manifest = render_manifest.get("source_phase_repair_manifest")
    if not isinstance(phase_manifest, dict):
        return []
    variants = phase_manifest.get("variants") if isinstance(phase_manifest.get("variants"), list) else []
    if not variants:
        return []
    candidates: list[tuple[str, Path]] = []
    base_minus_path = canonical_path(render_manifest.get("base_minus_path"))
    if base_minus_path.exists() and base_minus_path.is_file():
        candidates.append((candidate_id_from_path(base_minus_path), base_minus_path))
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        path = canonical_path(variant.get("path"))
        candidate_id = str(variant.get("variant") or candidate_id_from_path(path))
        candidates.append((candidate_id, path))
    return unique_existing_candidate_paths(candidates)


def minus_judge_context(source_manifest_path: Path, source_manifest: dict[str, Any]) -> dict[str, str]:
    result_judge_name = str(source_manifest.get("result_judge") or "")
    if not result_judge_name:
        return {}
    result_judge_path = source_manifest_path.parent / result_judge_name
    if not result_judge_path.exists():
        return {}
    for row in iter_jsonl(result_judge_path):
        mix_path = str(row.get("mix_path") or "")
        vocal_path = str(row.get("vocal_path") or "")
        if mix_path and vocal_path:
            return {"mix_path": mix_path, "vocal_path": vocal_path}
    return {}


def phase_repair_selection_context(
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    queue_item: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], list[str], str]:
    queue_id = str(queue_item.get("queue_item_id") or "")
    if source_manifest.get("schema") != "short_minus_phase_repair_selection_pack_manifest.v1":
        return {}, {}, [f"minus_render_manifest_missing:{queue_id}"], ""
    phase_manifest_path = canonical_path(source_manifest.get("source_phase_repair_manifest_json"))
    if not phase_manifest_path.exists() or not phase_manifest_path.is_file():
        return {}, {}, [f"minus_phase_repair_manifest_missing:{queue_id}"], ""
    phase_manifest = read_json(phase_manifest_path)
    failed = collect_true_flag_failures("minus_phase_repair_manifest", phase_manifest)
    source_minus_path = canonical_path(phase_manifest.get("source_minus_path"))
    if not source_minus_path.exists() or not source_minus_path.is_file():
        failed.append(f"minus_phase_source_minus_missing:{queue_id}")
    judge_context = {
        "mix_path": str(phase_manifest.get("source_mix_path") or ""),
        "vocal_path": str(phase_manifest.get("source_vocal_path") or ""),
    }
    variant_ids = [
        str(variant.get("variant") or candidate_id_from_path(canonical_path(variant.get("path"))))
        for variant in phase_manifest.get("variants") or []
        if isinstance(variant, dict)
    ]
    render_manifest = {
        "base_candidate_id": str(phase_manifest.get("source_candidate_id") or source_minus_path.stem),
        "base_minus_path": str(source_minus_path),
        "alternate_candidate_ids": variant_ids,
        "applied_windows": [],
        "track_id": str(phase_manifest.get("track_id") or ""),
        "source_lineage": "phase_repair_selection_manifest",
        "source_phase_repair_manifest_json": str(phase_manifest_path),
        "source_phase_repair_manifest": phase_manifest,
        "source_selection_manifest_json": str(source_manifest_path),
    }
    return render_manifest, judge_context, stable_unique(failed), "phase_repair_selection_manifest"


def manifest_reference_path(base_path: Path, value: Any) -> Path:
    if isinstance(value, dict):
        raw = str(value.get("path") or value.get("manifest_path") or "")
    else:
        raw = str(value or "")
    if not raw:
        return Path("")
    path = canonical_path(raw)
    return path if path.is_absolute() else base_path.parent / path


def no_reference_minus_preview_context(
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    queue_item: dict[str, Any],
) -> tuple[list[tuple[str, Path]], dict[str, str], dict[str, Any], list[str], str]:
    if source_manifest.get("manifest_type") != "no_reference_minus_listen_preview":
        return [], {}, {}, ["no_reference_minus_preview_schema_unmatched"], ""
    queue_id = str(queue_item.get("queue_item_id") or "")
    source_candidate_manifest_path = manifest_reference_path(source_manifest_path, source_manifest.get("source_candidate_manifest"))
    if not source_candidate_manifest_path.exists() or not source_candidate_manifest_path.is_file():
        return [], {}, {}, [f"no_reference_minus_source_candidate_manifest_missing:{queue_id}"], "no_reference_minus_preview_source_candidate_manifest"
    source_candidate_manifest = read_json(source_candidate_manifest_path)
    failed = collect_true_flag_failures("no_reference_minus_preview", source_manifest)
    failed.extend(collect_true_flag_failures("no_reference_minus_source_candidate_manifest", source_candidate_manifest))

    candidate_row = source_candidate_manifest.get("candidate_row") if isinstance(source_candidate_manifest.get("candidate_row"), dict) else {}
    base_minus_path = canonical_path(source_candidate_manifest.get("base_minus_path") or candidate_row.get("base_minus_path"))
    if not base_minus_path.exists() or not base_minus_path.is_file():
        failed.append(f"no_reference_minus_base_audio_missing:{queue_id}")

    selected_minus_path = canonical_path(
        candidate_row.get("selected_minus_path")
        or candidate_row.get("minus_path")
        or candidate_row.get("instrumental_path")
        or source_candidate_manifest.get("selected_minus_path")
        or source_candidate_manifest.get("minus_path")
        or source_candidate_manifest.get("instrumental_path")
        or source_manifest.get("selected_audio_path")
        or source_manifest.get("selected_minus_path")
        or source_manifest.get("minus_path")
        or source_manifest.get("instrumental_path")
    )
    if not selected_minus_path.exists() or not selected_minus_path.is_file():
        failed.append(f"no_reference_minus_selected_audio_missing:{queue_id}")

    repair_candidate_id = str(source_candidate_manifest.get("candidate_id") or candidate_row.get("candidate_id") or candidate_id_from_path(selected_minus_path))
    candidate_sources = unique_existing_candidate_paths(
        [
            ("base_minus", base_minus_path),
            (repair_candidate_id, selected_minus_path),
        ]
    )
    selected_windows = []
    repair = source_candidate_manifest.get("repair") if isinstance(source_candidate_manifest.get("repair"), dict) else {}
    for window in repair.get("selected_windows") or []:
        if not isinstance(window, dict):
            continue
        selected_windows.append(
            {
                "alternate_candidate_id": repair_candidate_id,
                "window_start_sec": window.get("window_start_sec", window.get("start_sec")),
                "window_duration_sec": window.get("window_duration_sec", window.get("duration_sec")),
            }
        )
    judge_context = {
        "mix_path": str(source_manifest.get("mix_path") or candidate_row.get("mix_path") or source_candidate_manifest.get("raw_audio_path") or ""),
        "vocal_path": str(
            source_manifest.get("vocal_path")
            or candidate_row.get("selected_vocal_path")
            or candidate_row.get("vocal_path")
            or source_candidate_manifest.get("primary_vocals_path")
            or ""
        ),
    }
    render_manifest = {
        "base_candidate_id": "base_minus",
        "base_minus_path": str(base_minus_path),
        "alternate_candidate_ids": [candidate_id for candidate_id, _ in candidate_sources],
        "applied_windows": selected_windows,
        "track_id": str(source_manifest.get("track_id") or source_candidate_manifest.get("track_id") or candidate_row.get("track_id") or ""),
        "source_candidate_manifest_json": str(source_candidate_manifest_path),
        "source_selection_manifest_json": str(source_manifest_path),
    }
    return candidate_sources, judge_context, render_manifest, stable_unique(failed), "no_reference_minus_preview_source_candidate_manifest"


def judge_guided_minus_leakage_repair_context(
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
) -> tuple[list[tuple[str, Path]], dict[str, str], dict[str, Any], list[str], str]:
    if source_manifest.get("mode") != "judge_guided_minus_vocal_leakage_repair":
        return [], {}, {}, ["judge_guided_leakage_repair_schema_unmatched"], ""
    failed = collect_true_flag_failures("judge_guided_leakage_repair", source_manifest)
    if source_manifest.get("candidate_only") is not True:
        failed.append("judge_guided_leakage_repair_not_candidate_only")
    if source_manifest.get("delivery_allowed") is not False:
        failed.append("judge_guided_leakage_repair_delivery_allowed_not_false")
    candidate_row = source_manifest.get("candidate_row") if isinstance(source_manifest.get("candidate_row"), dict) else {}
    repaired_minus_path = canonical_path(
        source_manifest.get("rendered_minus_candidate_path")
        or candidate_row.get("selected_minus_path")
        or candidate_row.get("minus_path")
        or candidate_row.get("instrumental_path")
    )
    if not repaired_minus_path.exists() or not repaired_minus_path.is_file():
        failed.append("judge_guided_leakage_repair_audio_missing")
    candidate_id = str(source_manifest.get("candidate_id") or candidate_row.get("candidate_id") or candidate_id_from_path(repaired_minus_path))
    base_minus_raw = str(source_manifest.get("base_minus_path") or candidate_row.get("base_minus_path") or "").strip()
    base_minus_path = canonical_path(base_minus_raw) if base_minus_raw else None
    if base_minus_path is None:
        failed.append("judge_guided_leakage_repair_base_minus_missing")
    elif not base_minus_path.exists() or not base_minus_path.is_file():
        failed.append("judge_guided_leakage_repair_base_minus_audio_missing")
    repair = source_manifest.get("repair") if isinstance(source_manifest.get("repair"), dict) else {}
    selected_windows = []
    for window in repair.get("selected_windows") or []:
        if not isinstance(window, dict):
            continue
        selected_windows.append(
            {
                "alternate_candidate_id": candidate_id,
                "window_start_sec": window.get("window_start_sec", window.get("start_sec")),
                "window_duration_sec": window.get("window_duration_sec", window.get("duration_sec")),
            }
        )
    judge_context = {
        "mix_path": str(source_manifest.get("raw_audio_path") or candidate_row.get("mix_path") or candidate_row.get("source_audio_path") or ""),
        "vocal_path": str(
            source_manifest.get("primary_vocals_path")
            or candidate_row.get("selected_vocal_path")
            or candidate_row.get("vocal_path")
            or ""
        ),
    }
    render_manifest = {
        "base_candidate_id": candidate_id,
        "base_minus_path": str(base_minus_path or ""),
        "source_minus_path": str(base_minus_path or ""),
        "alternate_candidate_ids": [candidate_id],
        "applied_windows": selected_windows,
        "track_id": str(source_manifest.get("track_id") or candidate_row.get("track_id") or ""),
        "source_judge_guided_manifest_json": str(source_manifest_path),
    }
    return (
        unique_existing_candidate_paths([(candidate_id, repaired_minus_path)]),
        judge_context,
        render_manifest,
        stable_unique(failed),
        "judge_guided_minus_vocal_leakage_repair",
    )


def unique_existing_candidate_paths(candidates: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for candidate_id, path in candidates:
        key = str(path)
        if not candidate_id or key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        result.append((candidate_id, path))
    return result


def full_track_switch_context(
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
) -> tuple[list[tuple[str, Path]], dict[str, str], dict[str, Any], list[str], str]:
    if source_manifest.get("schema") != "full_track_switch_listen_pack.v1":
        return [], {}, {}, ["full_track_switch_schema_unmatched"], ""
    switch_manifest_path = canonical_path(source_manifest.get("switch_manifest_json"))
    if not switch_manifest_path.exists() or not switch_manifest_path.is_file():
        for row in source_manifest.get("listen_audio_files") or []:
            if isinstance(row, dict) and row.get("source_switch_manifest_json"):
                switch_manifest_path = canonical_path(row.get("source_switch_manifest_json"))
                break
    if not switch_manifest_path.exists() or not switch_manifest_path.is_file():
        return [], {}, {}, ["full_track_switch_manifest_missing"], "full_track_switch_manifest"
    switch_manifest = read_json(switch_manifest_path)
    failed = collect_true_flag_failures("full_track_switch_manifest", switch_manifest)
    candidates: list[tuple[str, Path]] = []
    rendered_minus = canonical_path(switch_manifest.get("rendered_minus_candidate_path"))
    if rendered_minus.exists():
        candidates.append(("fts", rendered_minus))
    selected_windows = switch_manifest.get("selected_windows") if isinstance(switch_manifest.get("selected_windows"), list) else []
    for window in selected_windows:
        if not isinstance(window, dict):
            continue
        minus_path = canonical_path(window.get("selected_minus_path"))
        candidate_id = str(window.get("selected_candidate_id") or window.get("candidate_id") or minus_path.stem)
        candidates.append((candidate_id, minus_path))
    vocal_path = str(switch_manifest.get("rendered_vocal_candidate_path") or "")
    if not vocal_path:
        for window in selected_windows:
            if isinstance(window, dict) and window.get("selected_vocal_path"):
                vocal_path = str(window.get("selected_vocal_path") or "")
                break
    candidate_row = switch_manifest.get("candidate_row") if isinstance(switch_manifest.get("candidate_row"), dict) else {}
    source_minus_raw = str(
        switch_manifest.get("source_minus_path")
        or switch_manifest.get("base_minus_path")
        or candidate_row.get("source_minus_path")
        or candidate_row.get("base_minus_path")
        or ""
    ).strip()
    source_minus_path = canonical_path(source_minus_raw) if source_minus_raw else None
    if source_minus_path is None:
        failed.append("full_track_switch_base_minus_missing")
    elif not source_minus_path.exists() or not source_minus_path.is_file():
        failed.append("full_track_switch_base_minus_audio_missing")
    judge_context = {
        "mix_path": str(switch_manifest.get("mix_path") or candidate_row.get("mix_path") or ""),
        "vocal_path": vocal_path,
    }
    render_manifest = {
        "base_candidate_id": str(switch_manifest.get("base_candidate_id") or ""),
        "base_minus_path": str(source_minus_path or ""),
        "source_minus_path": str(source_minus_path or ""),
        "alternate_candidate_ids": [candidate_id for candidate_id, _ in candidates],
        "applied_windows": [
            {
                "alternate_candidate_id": str(window.get("selected_candidate_id") or window.get("candidate_id") or ""),
                "window_start_sec": window.get("window_start_sec"),
                "window_duration_sec": window.get("window_duration_sec"),
            }
            for window in selected_windows
            if isinstance(window, dict)
        ],
        "source_switch_manifest_json": str(switch_manifest_path),
        "source_selection_manifest_json": str(source_manifest_path),
    }
    return unique_existing_candidate_paths(candidates), judge_context, render_manifest, stable_unique(failed), "full_track_switch_manifest"


def stem_audio_pack_context(
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
) -> tuple[list[tuple[str, Path]], dict[str, str], dict[str, Any], list[str], str]:
    if source_manifest.get("schema") != "stem_manifest_audio_pack_manifest.v1":
        return [], {}, {}, ["stem_audio_pack_schema_unmatched"], ""
    stems = source_manifest.get("stems") if isinstance(source_manifest.get("stems"), list) else []
    minus_stem = next((row for row in stems if isinstance(row, dict) and str(row.get("stem_type") or "") == "minus"), None)
    vocal_stem = next((row for row in stems if isinstance(row, dict) and str(row.get("stem_type") or "") == "vocal"), None)
    if not isinstance(minus_stem, dict):
        return [], {}, {}, ["stem_audio_pack_minus_missing"], ""
    minus_path = canonical_path(minus_stem.get("listen_audio_path") or minus_stem.get("source_audio_path"))
    vocal_path = canonical_path(vocal_stem.get("listen_audio_path") or vocal_stem.get("source_audio_path")) if isinstance(vocal_stem, dict) else Path("")
    failed = []
    if not minus_path.exists() or not minus_path.is_file():
        failed.append("stem_audio_pack_minus_audio_missing")
    if not vocal_path.exists() or not vocal_path.is_file():
        failed.append("stem_audio_pack_vocal_audio_missing")
    judge_context = {
        "mix_path": str(source_manifest.get("source_audio_path") or ""),
        "vocal_path": str(vocal_path),
    }
    render_manifest = {
        "base_candidate_id": "source_minus",
        "base_minus_path": str(minus_path),
        "alternate_candidate_ids": ["source_minus"],
        "applied_windows": [],
        "source_selection_manifest_json": str(source_manifest_path),
    }
    return unique_existing_candidate_paths([("source_minus", minus_path)]), judge_context, render_manifest, stable_unique(failed), "stem_manifest_audio_pack"


def applied_window_count(candidate_id: str, render_manifest: dict[str, Any], queue_item: dict[str, Any]) -> int:
    manifest_windows = render_manifest.get("applied_windows") if isinstance(render_manifest.get("applied_windows"), list) else []
    queue_context = queue_item.get("diagnostic_context") if isinstance(queue_item.get("diagnostic_context"), dict) else {}
    queue_windows = queue_context.get("target_windows") if isinstance(queue_context.get("target_windows"), list) else []
    all_windows = [*manifest_windows, *queue_windows]
    matched: set[tuple[str, float, float]] = set()
    for window in all_windows:
        if not isinstance(window, dict) or str(window.get("alternate_candidate_id") or "") != candidate_id:
            continue
        start = float(window.get("window_start_sec", window.get("start_sec", 0.0)) or 0.0)
        duration = float(window.get("window_duration_sec", window.get("duration_sec", 0.0)) or 0.0)
        matched.add((candidate_id, round(start, 6), round(duration, 6)))
    return len(matched)


def build_minus_candidates(
    *,
    queue_item: dict[str, Any],
    output_dir: Path,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    source_manifest_path = minus_source_manifest_path(queue_item)
    if not source_manifest_path.exists():
        return [], [f"minus_source_manifest_missing:{queue_item.get('queue_item_id')}"]
    source_manifest = read_json(source_manifest_path)
    judge_context = minus_judge_context(source_manifest_path, source_manifest)
    render_manifest_name = str(source_manifest.get("render_manifest") or "")
    render_manifest_path = source_manifest_path.parent / render_manifest_name
    source_lineage = "render_manifest"
    candidate_sources: list[tuple[str, Path]] = []
    if render_manifest_name and render_manifest_path.exists() and render_manifest_path.is_file():
        render_manifest = read_json(render_manifest_path)
        base_minus_path = canonical_path(render_manifest.get("base_minus_path"))
        candidate_dir = base_minus_path.parent
        if not candidate_dir.exists():
            return [], [f"minus_candidate_dir_missing:{queue_item.get('queue_item_id')}"]
        candidate_sources = [
            (candidate_id_from_path(path), path)
            for path in sorted(candidate_dir.glob("c*.flac") if candidate_dir.exists() else [])
            if path.is_file()
        ]
    else:
        render_manifest, recovered_judge_context, recovery_failed, recovered_lineage = phase_repair_selection_context(
            source_manifest_path,
            source_manifest,
            queue_item,
        )
        if recovery_failed:
            nr_candidates, nr_judge_context, nr_render_manifest, nr_failed, nr_lineage = no_reference_minus_preview_context(
                source_manifest_path,
                source_manifest,
                queue_item,
            )
            if nr_lineage:
                if nr_failed:
                    return [], nr_failed
                render_manifest = nr_render_manifest
                judge_context = nr_judge_context
                source_lineage = nr_lineage
                candidate_sources = nr_candidates
            else:
                jgl_candidates, jgl_judge_context, jgl_render_manifest, jgl_failed, jgl_lineage = (
                    judge_guided_minus_leakage_repair_context(
                        source_manifest_path,
                        source_manifest,
                    )
                )
                if jgl_lineage:
                    if jgl_failed:
                        return [], jgl_failed
                    render_manifest = jgl_render_manifest
                    judge_context = jgl_judge_context
                    source_lineage = jgl_lineage
                    candidate_sources = jgl_candidates
                else:
                    fts_candidates, fts_judge_context, fts_render_manifest, fts_failed, fts_lineage = full_track_switch_context(
                        source_manifest_path,
                        source_manifest,
                    )
                    if fts_lineage:
                        if fts_failed:
                            return [], fts_failed
                        render_manifest = fts_render_manifest
                        judge_context = fts_judge_context
                        source_lineage = fts_lineage
                        candidate_sources = fts_candidates
                    else:
                        stem_candidates, stem_judge_context, stem_render_manifest, stem_failed, stem_lineage = stem_audio_pack_context(
                            source_manifest_path,
                            source_manifest,
                        )
                        if stem_failed:
                            return [], recovery_failed
                        render_manifest = stem_render_manifest
                        judge_context = stem_judge_context
                        source_lineage = stem_lineage
                        candidate_sources = stem_candidates
        else:
            base_minus_path = canonical_path(render_manifest.get("base_minus_path"))
            candidate_dir = base_minus_path.parent
            if not candidate_dir.exists():
                return [], [f"minus_candidate_dir_missing:{queue_item.get('queue_item_id')}"]
            candidate_sources = phase_variant_candidate_sources(render_manifest)
            if not candidate_sources:
                candidate_sources = [
                    (candidate_id_from_path(path), path)
                    for path in sorted(candidate_dir.glob("c*.flac") if candidate_dir.exists() else [])
                    if path.is_file()
                ]
            judge_context = recovered_judge_context
            source_lineage = recovered_lineage
    failed = collect_true_flag_failures("minus_source_manifest", source_manifest)
    failed.extend(collect_true_flag_failures("minus_render_manifest", render_manifest))

    candidate_sources = unique_existing_candidate_paths(candidate_sources)[: max(0, max_candidates)]
    rows: list[dict[str, Any]] = []
    queue_item_id = str(queue_item.get("queue_item_id") or "q000")
    validate_artifact_segment(queue_item_id)
    for index, (minus_candidate_id, candidate_path) in enumerate(candidate_sources, start=1):
        filename = indexed_audio_filename("m", index)
        listen_path = output_dir / ARTIFACT_AUDIO_DIR / queue_item_id / filename
        listen_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, listen_path)
        metadata = audio_metadata(listen_path)
        rows.append(
            {
                "schema": "short_listen_repair_candidate.v1",
                "queue_item_id": queue_item_id,
                "route_intent": "minus_repair",
                "stem_type": "minus",
                "artifact_alias": filename.removesuffix(".flac"),
                "artifact_filename": filename,
                "artifact_relative_path": str(listen_path.relative_to(output_dir).as_posix()),
                "listen_audio_path": str(listen_path),
                "listen_audio_sha256": sha256_file(listen_path),
                "sample_rate": metadata["sample_rate"],
                "channels": metadata["channels"],
                "duration_sec": metadata["duration_sec"],
                "source_audio_path": str(candidate_path),
                "source_minus_path": str(render_manifest.get("source_minus_path") or render_manifest.get("base_minus_path") or ""),
                "base_minus_path": str(render_manifest.get("base_minus_path") or ""),
                "mix_path": judge_context.get("mix_path", ""),
                "vocal_path": judge_context.get("vocal_path", ""),
                "instrumental_path": str(listen_path),
                "candidate_id": f"{queue_item_id}_{minus_candidate_id}",
                "track_id": str(source_manifest.get("track_id") or render_manifest.get("track_id") or source_track_id(queue_item) or ""),
                "source_manifest": str(source_manifest_path),
                "render_manifest": str(render_manifest_path) if source_lineage == "render_manifest" else "",
                "source_lineage": source_lineage,
                "minus_candidate_id": minus_candidate_id,
                "base_candidate_id": str(render_manifest.get("base_candidate_id") or ""),
                "alternate_candidate_ids": [str(value) for value in render_manifest.get("alternate_candidate_ids") or []],
                "applied_window_count": applied_window_count(minus_candidate_id, render_manifest, queue_item),
                "quality_status": "repair_candidate_no_reference_scored_source",
                "delivery_status": "accepted",
                "delivery_allowed": True,
                "presentation_status": "accepted_for_repair_listen_pack",
                "release_status": "listen_preview_only",
                "release_ready": False,
                "can_publish_as_result": False,
                "semantic_quality_verified": False,
            }
        )
    if not rows:
        failed.append(f"no_minus_repair_candidates:{queue_item_id}")
    return rows, stable_unique(failed)


def route_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("route_intent") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    content_key = str(row.get("source_audio_sha256") or row.get("listen_audio_sha256") or row.get("source_audio_path") or "")
    return (
        str(row.get("route_intent") or ""),
        str(row.get("stem_type") or ""),
        str(row.get("source_lineage") or row.get("source_candidate_family") or ""),
        content_key,
    )


def deduplicate_candidates(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    deduplicated = 0
    for row in rows:
        key = dedupe_key(row)
        if key[3] and key in seen:
            deduplicated += 1
            listen_path = Path(str(row.get("listen_audio_path") or ""))
            if listen_path.exists() and listen_path.is_file():
                listen_path.unlink()
            continue
        seen.add(key)
        result.append(row)
    return result, deduplicated


def minus_judge_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("route_intent") != "minus_repair":
            continue
        mix_path = str(row.get("mix_path") or "")
        vocal_path = str(row.get("vocal_path") or "")
        instrumental_path = str(row.get("instrumental_path") or "")
        if not mix_path or not vocal_path or not instrumental_path:
            continue
        result.append(
            {
                "track_id": str(row.get("track_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "mix_path": mix_path,
                "vocal_path": vocal_path,
                "instrumental_path": instrumental_path,
                "minus_path": instrumental_path,
                "selected_minus_path": instrumental_path,
                "source_manifest": str(row.get("source_manifest") or ""),
                "source_candidate_id": str(row.get("minus_candidate_id") or ""),
                "can_publish_as_result": False,
                "can_mark_release_ready": False,
            }
        )
    return result


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short listen repair candidate pack",
        "",
        f"- delivery_status: `{manifest['delivery_status']}`",
        f"- candidate_count: `{manifest['candidate_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Candidates",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['queue_item_id']}` `{row['artifact_relative_path']}` `{row['route_intent']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_short_listen_repair_candidate_pack(
    *,
    repair_queue_manifest_json: Path,
    output_dir: Path,
    vocal_training_examples_jsonl: Path | None = None,
    max_vocal_candidates: int = 6,
    max_minus_candidates: int = 9,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", repair_queue_manifest_json=str(repair_queue_manifest_json))

    queue_manifest = read_json(repair_queue_manifest_json)
    failed: list[str] = collect_true_flag_failures("repair_queue", queue_manifest)
    if queue_manifest.get("schema") != "short_listen_repair_queue_manifest.v1":
        failed.append("repair_queue_schema_unexpected")
    queue_status = str(queue_manifest.get("queue_status") or "")
    queue_source = str(queue_manifest.get("queue_source") or "")
    if not queue_status_is_ready(queue_status, queue_source):
        failed.append("repair_queue_not_ready")
    if queue_manifest.get("failed_checks") not in ([], None):
        failed.append("repair_queue_failed_checks_not_empty")
    rows_jsonl = canonical_path(queue_manifest.get("rows_jsonl"))
    if not rows_jsonl.exists():
        failed.append("repair_queue_rows_missing")

    selected: list[dict[str, Any]] = []
    if not failed:
        queue_rows = list(iter_jsonl(rows_jsonl))
        for queue_index, queue_item in enumerate(queue_rows, start=1):
            progress(
                progress_path,
                "queue_item_start",
                queue_index=queue_index,
                queue_item_id=str(queue_item.get("queue_item_id") or ""),
                route_intent=str(queue_item.get("route_intent") or ""),
            )
            failed.extend(collect_true_flag_failures("repair_queue_item", queue_item))
            intent = str(queue_item.get("route_intent") or "")
            if intent == "vocal_candidate_replacement":
                if vocal_training_examples_jsonl is None:
                    item_rows, item_failed = [], [f"vocal_training_examples_required:{queue_item.get('queue_item_id')}"]
                else:
                    item_rows, item_failed = build_vocal_candidates(
                        queue_item=queue_item,
                        examples_path=vocal_training_examples_jsonl,
                        output_dir=output_dir,
                        max_candidates=max_vocal_candidates,
                    )
            elif intent == "minus_repair":
                item_rows, item_failed = build_minus_candidates(
                    queue_item=queue_item,
                    output_dir=output_dir,
                    max_candidates=max_minus_candidates,
                )
            else:
                item_rows, item_failed = [], [f"unsupported_route_intent:{intent}"]
            failed.extend(item_failed)
            selected.extend(item_rows)
            progress(
                progress_path,
                "queue_item_finish",
                queue_item_id=str(queue_item.get("queue_item_id") or ""),
                candidate_count=len(item_rows),
                failed_checks=item_failed,
            )

    failed = stable_unique(failed)
    selected, deduplicated_candidate_count = deduplicate_candidates(selected)
    accepted = bool(selected) and not failed
    if not accepted:
        selected = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
    judge_rows = minus_judge_rows(selected) if accepted else []

    manifest = {
        "schema": "short_listen_repair_candidate_pack_manifest.v1",
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_repair_listen_pack" if accepted else "blocked_from_repair_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "deliverable": False,
        "listenable": accepted,
        "candidate_count": len(selected),
        "deduplicated_candidate_count": deduplicated_candidate_count,
        "route_intent_counts": route_counts(selected),
        "repair_queue_manifest_json": str(repair_queue_manifest_json),
        "queue_source": queue_source,
        "vocal_training_examples_jsonl": str(vocal_training_examples_jsonl or ""),
        "failed_checks": failed,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "minus_judge_candidates_jsonl": str(output_dir / MINUS_JUDGE_NAME),
        "minus_judge_candidate_count": len(judge_rows),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "queue_dir_strategy": "qNNN",
            "filename_strategy": "indexed_candidate_alias",
            "raw_track_or_candidate_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "contract": {
            "audio_copied_without_modification": True,
            "candidate_mining_only": True,
            "no_inference_run": True,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "release_ready_forced_false": True,
            "short_windows_safe_artifact_names": True,
            "dirty_stems_never_ground_truth": True,
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "not_claimed": ["release_ready", "publishable_result", "audio_repaired", "training_started", "semantic_ground_truth"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected)
    write_jsonl(output_dir / MINUS_JUDGE_NAME, judge_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, selected), encoding="utf-8")
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], candidate_count=len(selected), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short repair-candidate listen pack from a repair queue.")
    parser.add_argument("--repair-queue-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocal-training-examples-jsonl", type=Path)
    parser.add_argument("--max-vocal-candidates", type=int, default=6)
    parser.add_argument("--max-minus-candidates", type=int, default=9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_repair_candidate_pack.py",
        entrypoint_path="/app/tools/build_short_listen_repair_candidate_pack.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_repair_candidate_pack(
        repair_queue_manifest_json=args.repair_queue_manifest_json,
        output_dir=args.output_dir,
        vocal_training_examples_jsonl=args.vocal_training_examples_jsonl,
        max_vocal_candidates=int(args.max_vocal_candidates),
        max_minus_candidates=int(args.max_minus_candidates),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "candidate_count": manifest["candidate_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
