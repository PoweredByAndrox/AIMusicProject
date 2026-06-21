from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audio_quality_critic_common import (
    BLOCKED_FILE,
    MANIFEST_FILE,
    PROGRESS_FILE,
    PROMOTION_LABEL_TYPES,
    SAFETY_FLAGS,
    TARGET_NAMES,
    append_progress,
    as_list,
    iter_jsonl,
    load_rag_current,
    normalize_stem_type,
    read_json,
    stable_hash,
    write_json,
    write_jsonl,
)
from tools.docker_runtime import require_docker_runtime


ROWS_FILE = "rows.jsonl"
TENSOR_DIR = "tensors"
MANIFEST_SCHEMA = "audio_quality_verified_rows_manifest.v1"
VERIFIED_CORPUS_SCHEMA = "verified_audio_training_corpus_manifest.v1"
GENERIC_EXAMPLE_SCHEMA = "generic_finetuning_audio_training_example.v1"
VERIFIED_REPORT_SCHEMA = "verified_audio_training_corpus_row.v1"
ROUTE_ID = "verified_audio_training_corpus_intake"
TRAINABLE_PERMISSION_STATUSES = {"owned", "licensed", "cleared", "project_owned"}
CRITICAL_GROUND_TRUTH_FLAGS = (
    "generated_stem_claims_ground_truth",
    "pseudo_teacher_targets_are_ground_truth",
)
BLOCKED_TRUE_FLAGS = (
    "delivery_allowed",
    "release_ready",
    "can_publish_as_result",
    "thresholds_relaxed",
    "model_registry_edited",
    "runtime_gates_changed",
)


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    return path if path.is_absolute() else base_dir / path


def _normalize_split(value: Any) -> str:
    split = str(value or "train").strip().lower()
    if split in {"holdout", "locked-holdout", "locked_holdout"}:
        return "locked_holdout"
    if split in {"validation", "valid", "val"}:
        return "dev"
    if split in {"train", "dev", "test", "locked_holdout"}:
        return split
    return "train"


def _verified_report_map(row_report_jsonl: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    if not row_report_jsonl.exists():
        return reports
    for report in iter_jsonl(row_report_jsonl):
        row_id = str(report.get("id") or "").strip()
        if row_id:
            reports[row_id] = report
    return reports


def _manifest_flag_reasons(prefix: str, payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in BLOCKED_TRUE_FLAGS:
        if payload.get(key) is True:
            reasons.append(f"{prefix}_{key}_true")
    safety = payload.get("safety")
    if isinstance(safety, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if safety.get(key) is True:
                reasons.append(f"{prefix}_safety_{key}_true")
    return reasons


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _report_mismatch_reasons(row: dict[str, Any], report: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("track_id", "source_audio_path", "target_audio_path"):
        if _normalized_text(report.get(key)) != _normalized_text(row.get(key)):
            reasons.append(f"verified_row_report_{key}_mismatch")
    if normalize_stem_type(report.get("stem_type")) != normalize_stem_type(row.get("stem_type")):
        reasons.append("verified_row_report_stem_type_mismatch")
    if bool(report.get("human_verified")) != bool(row.get("human_verified")):
        reasons.append("verified_row_report_human_verified_mismatch")
    if bool(report.get("reference_verified")) != bool(row.get("reference_verified")):
        reasons.append("verified_row_report_reference_verified_mismatch")
    return reasons


def _source_lock_reasons(source_lock_pack_json: Path, *, rag_current_json: Path) -> list[str]:
    reasons: list[str] = []
    if not str(source_lock_pack_json) or not source_lock_pack_json.exists():
        return ["source_lock_pack_json_missing"]
    try:
        source_lock = read_json(source_lock_pack_json)
    except Exception as exc:  # noqa: BLE001
        return ["source_lock_pack_json_read_failed:" + type(exc).__name__]
    if source_lock.get("schema") != "audio_quality_source_lock.v1":
        reasons.append("source_lock_schema_unexpected")
    reasons.extend(_manifest_flag_reasons("source_lock", source_lock))
    if source_lock.get("rag_idx_exists") is not True:
        reasons.append("source_lock_rag_idx_exists_not_true")
    rag = load_rag_current(rag_current_json)
    lock_rag_id = str(source_lock.get("current_rag_id") or "").strip()
    if not lock_rag_id:
        reasons.append("source_lock_current_rag_missing")
    elif lock_rag_id != str(rag.get("rag_id") or ""):
        reasons.append("source_lock_current_rag_mismatch")
    return _unique(reasons)


def _critical_ground_truth_claims(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in CRITICAL_GROUND_TRUTH_FLAGS:
        if row.get(key) is True:
            reasons.append(key)
    safety = row.get("safety")
    if isinstance(safety, dict):
        for key in CRITICAL_GROUND_TRUTH_FLAGS:
            if safety.get(key) is True:
                reasons.append(key)
    return _unique(reasons)


def _row_block_reasons(row: dict[str, Any], report: dict[str, Any] | None, *, source_lock_pack_json: Path) -> list[str]:
    reasons: list[str] = []
    stem = normalize_stem_type(row.get("stem_type"))
    permission_status = str(row.get("permission_status") or "").strip().lower()
    source_audio_text = str(row.get("source_audio_path") or "").strip()
    target_audio_text = str(row.get("target_audio_path") or "").strip()
    source_audio_path = Path(source_audio_text) if source_audio_text else Path("__missing_source_audio__")
    target_audio_path = Path(target_audio_text) if target_audio_text else Path("__missing_target_audio__")
    if row.get("schema") != GENERIC_EXAMPLE_SCHEMA:
        reasons.append("unexpected_example_schema")
    if report is None:
        reasons.append("missing_verified_row_report")
    else:
        if report.get("schema") not in {VERIFIED_REPORT_SCHEMA, None}:
            reasons.append("unexpected_verified_row_report_schema")
        if report.get("safe_for_training") is not True:
            reasons.append("verified_row_report_unsafe")
            reasons.extend(as_list(report.get("unsafe_reasons")))
        reasons.extend(_report_mismatch_reasons(row, report))
    if row.get("trainable") is not True:
        reasons.append("not_marked_trainable")
    if permission_status not in TRAINABLE_PERMISSION_STATUSES:
        reasons.append("permission_not_trainable")
    if row.get("human_verified") is not True and row.get("reference_verified") is not True:
        reasons.append("not_human_or_reference_verified")
    if row.get("delivery_allowed") is True:
        reasons.append("input_row_delivery_allowed_true")
    if row.get("release_ready") is True:
        reasons.append("input_row_release_ready_true")
    if stem not in {"vocal", "minus"}:
        reasons.append("unsupported_aqc_stem_type")
    if not str(row.get("id") or "").strip():
        reasons.append("missing_id")
    if not str(row.get("track_id") or "").strip():
        reasons.append("missing_track_id")
    if not source_audio_text or not source_audio_path.exists():
        reasons.append("source_audio_not_found")
    if not target_audio_text or not target_audio_path.exists():
        reasons.append("target_audio_not_found")
    if not str(source_lock_pack_json) or not source_lock_pack_json.exists():
        reasons.append("source_lock_pack_json_missing")
    return _unique(reasons)


def _envelope(samples: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.size == 0:
        zeros = np.zeros((bins,), dtype=np.float32)
        return zeros, zeros, 0.0
    mono = samples.astype(np.float32).mean(axis=1)
    chunks = np.array_split(mono, bins)
    rms = np.asarray([math.sqrt(float(np.mean(np.square(chunk)))) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    peak = np.asarray([float(np.max(np.abs(chunk))) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    corr = 1.0
    if samples.shape[1] >= 2 and samples.shape[0] > 1:
        left = samples[:, 0].astype(np.float32)
        right = samples[:, 1].astype(np.float32)
        if float(np.std(left)) > 1.0e-8 and float(np.std(right)) > 1.0e-8:
            corr = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(corr):
        corr = 0.0
    return rms, peak, float(corr)


def _read_audio_tensor(path: Path, *, window_sec: float, tensor_bins: int) -> tuple[np.ndarray, float]:
    import soundfile as sf

    info = sf.info(str(path))
    frames = max(1, min(int(info.frames), int(float(window_sec) * int(info.samplerate))))
    samples, _sample_rate = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
    rms, peak, corr = _envelope(np.asarray(samples, dtype=np.float32), tensor_bins)
    return np.stack([rms, peak], axis=0).astype(np.float32), corr


def _write_tensor(
    *,
    row: dict[str, Any],
    output_dir: Path,
    row_index: int,
    window_sec: float,
    tensor_bins: int,
) -> tuple[str, str]:
    stem = normalize_stem_type(row.get("stem_type"))
    try:
        mix_tensor, mix_corr = _read_audio_tensor(Path(str(row.get("source_audio_path") or "")), window_sec=window_sec, tensor_bins=tensor_bins)
        target_tensor, target_corr = _read_audio_tensor(Path(str(row.get("target_audio_path") or "")), window_sec=window_sec, tensor_bins=tensor_bins)
    except Exception as exc:  # noqa: BLE001
        return "", "tensor_extract_failed:" + type(exc).__name__
    zeros = np.zeros_like(target_tensor, dtype=np.float32)
    vocal_tensor = target_tensor if stem == "vocal" else zeros
    minus_tensor = target_tensor if stem == "minus" else zeros
    tensor_path = output_dir / TENSOR_DIR / f"q{row_index:03d}.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tensor_path,
        mix=mix_tensor,
        vocal=vocal_tensor,
        minus=minus_tensor,
        stereo_corr=np.asarray([mix_corr, target_corr, 0.0 if stem == "vocal" else target_corr], dtype=np.float32),
    )
    return str(tensor_path), ""


def _targets() -> dict[str, float]:
    payload = {name: 0.0 for name in TARGET_NAMES}
    payload["accept_probability"] = 1.0
    payload["uncertainty"] = 0.0
    return payload


def _path_key(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    candidate = Path(text)
    try:
        if candidate.exists():
            text = str(candidate.resolve())
    except OSError:
        pass
    return text.replace("\\", "/").lower()


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _rag_ref_candidates(ref: dict[str, Any], roots_by_alias: dict[str, dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for key in ("original_path", "source_manifest_json"):
        value = str(ref.get(key) or "").strip()
        if value:
            candidates.append(value)
    root_alias = str(ref.get("root_alias") or "").strip()
    relative_path = str(ref.get("relative_path") or "").strip()
    root = roots_by_alias.get(root_alias)
    if root and relative_path:
        original_path = str(root.get("original_path") or "").strip()
        if original_path:
            candidates.append(str(Path(original_path) / relative_path))
    return candidates


def _rag_refs(rag_current_json: Path, verified_corpus_manifest_json: Path) -> list[dict[str, Any]]:
    rag = load_rag_current(rag_current_json)
    if not rag.get("idx_exists"):
        return []
    idx_path = rag_current_json.parent / str(rag.get("idx") or "")
    try:
        idx = read_json(idx_path)
    except Exception:  # noqa: BLE001
        return []
    outputs = idx.get("outputs") if isinstance(idx.get("outputs"), dict) else {}
    idx_dir = idx_path.parent
    roots_path = idx_dir / str(outputs.get("roots") or "roots.jsonl")
    refs_path = idx_dir / str(outputs.get("manifest_refs") or "manifest_refs.jsonl")
    roots_by_alias = {str(row.get("root_alias") or ""): row for row in _load_jsonl_rows(roots_path) if row.get("root_alias")}
    target = _path_key(verified_corpus_manifest_json)
    refs: list[dict[str, Any]] = []
    for ref in _load_jsonl_rows(refs_path):
        if target not in {_path_key(candidate) for candidate in _rag_ref_candidates(ref, roots_by_alias)}:
            continue
        if not str(ref.get("chunk_id") or "").strip() or not str(ref.get("sha256") or "").strip():
            continue
        refs.append(
            {
                "rag_id": str(rag.get("rag_id") or ""),
                "idx": str(rag.get("idx") or ""),
                "idx_exists": True,
                "ref_type": "verified_corpus_manifest_chunk_ref",
                "root": str(ref.get("root") or ""),
                "root_alias": str(ref.get("root_alias") or ""),
                "relative_path": str(ref.get("relative_path") or ""),
                "chunk_id": str(ref.get("chunk_id") or ""),
                "sha256": str(ref.get("sha256") or ""),
                "zip_member": str(ref.get("zip_member") or ""),
                "source_manifest_json": str(verified_corpus_manifest_json),
            }
        )
    return refs


def _source_manifest_refs(
    *,
    verified_corpus_manifest_json: Path,
    audio_training_examples_jsonl: Path,
    row_report_jsonl: Path,
    row: dict[str, Any],
) -> list[str]:
    return _unique(
        [
            str(verified_corpus_manifest_json),
            str(audio_training_examples_jsonl),
            str(row_report_jsonl),
            str(row.get("manifest_path") or ""),
        ]
    )


def _aqc_row(
    *,
    row: dict[str, Any],
    tensor_path: str,
    row_index: int,
    verified_corpus_manifest_json: Path,
    audio_training_examples_jsonl: Path,
    row_report_jsonl: Path,
    rag_refs: list[dict[str, Any]],
    source_lock_pack_json: Path,
    window_sec: float,
) -> dict[str, Any]:
    label_type = "reference_verified" if row.get("reference_verified") is True else "human_verified"
    if label_type not in PROMOTION_LABEL_TYPES:
        raise ValueError(f"unexpected_promotion_label_type:{label_type}")
    row_id = str(row.get("id") or f"v{row_index:03d}")
    return {
        "track_id": str(row.get("track_id") or ""),
        "candidate_id": "ver_" + stable_hash([row_id, row.get("track_id"), row.get("stem_type")])[:12],
        "stem_type": normalize_stem_type(row.get("stem_type")),
        "window_start_sec": 0.0,
        "window_end_sec": float(window_sec),
        "mix_path": str(row.get("source_audio_path") or ""),
        "vocal_path": str(row.get("target_audio_path") or "") if normalize_stem_type(row.get("stem_type")) == "vocal" else "",
        "minus_path": str(row.get("target_audio_path") or "") if normalize_stem_type(row.get("stem_type")) == "minus" else "",
        "tensor_path": tensor_path,
        "features": None,
        "route_id": ROUTE_ID,
        "label_type": label_type,
        "label": 1,
        "label_confidence": 1.0 if label_type == "reference_verified" else 0.95,
        "split": _normalize_split(row.get("split")),
        "source_manifest_refs": _source_manifest_refs(
            verified_corpus_manifest_json=verified_corpus_manifest_json,
            audio_training_examples_jsonl=audio_training_examples_jsonl,
            row_report_jsonl=row_report_jsonl,
            row=row,
        ),
        "rag_refs": rag_refs,
        "source_lock_pack_json": str(source_lock_pack_json),
        "issue_tags": [],
        "blocker_reasons": [],
        "targets": _targets(),
        "generated_stem_claims_ground_truth": False,
        "pseudo_teacher_targets_are_ground_truth": False,
        "delivery_allowed": False,
        "release_ready": False,
    }


def _blocked_row(row: dict[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "track_id": str(row.get("track_id") or ""),
        "stem_type": normalize_stem_type(row.get("stem_type")),
        "blocked_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "delivery_allowed": False,
        "release_ready": False,
    }


def _write_outputs(
    *,
    output_dir: Path,
    progress_path: Path,
    verified_corpus_manifest_json: Path,
    audio_training_examples_jsonl: Path,
    row_report_jsonl: Path,
    rag_current_json: Path,
    source_lock_pack_json: Path,
    rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    errors: list[str],
    start_time: float,
) -> dict[str, Any]:
    write_jsonl(output_dir / ROWS_FILE, rows)
    write_jsonl(output_dir / BLOCKED_FILE, blocked)
    label_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows:
        label_counts[str(row.get("label_type") or "")] = label_counts.get(str(row.get("label_type") or ""), 0) + 1
        split_counts[str(row.get("split") or "")] = split_counts.get(str(row.get("split") or ""), 0) + 1
    rag = load_rag_current(rag_current_json)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_type": "audio_quality_verified_rows",
        "build_status": "passed" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "verified_corpus_manifest_json": str(verified_corpus_manifest_json),
        "audio_training_examples_jsonl": str(audio_training_examples_jsonl),
        "row_report_jsonl": str(row_report_jsonl),
        "rag_current_json": str(rag_current_json),
        "current_rag_id": str(rag.get("rag_id") or ""),
        "rag_idx": str(rag.get("idx") or ""),
        "rag_idx_exists": bool(rag.get("idx_exists")),
        "source_lock_pack_json": str(source_lock_pack_json),
        "row_count": len(rows),
        "blocked_row_count": len(blocked),
        "tensor_row_count": sum(1 for row in rows if row.get("tensor_path")),
        "feature_only_row_count": sum(1 for row in rows if not row.get("tensor_path")),
        "label_type_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "outputs": {
            "rows_jsonl": str(output_dir / ROWS_FILE),
            "blocked_jsonl": str(output_dir / BLOCKED_FILE),
            "manifest_json": str(output_dir / MANIFEST_FILE),
            "progress_jsonl": str(progress_path),
            "tensor_dir": str(output_dir / TENSOR_DIR),
        },
        "presentation_status": "audio_quality_verified_rows_not_for_listen_pack",
        "training_started": False,
        "promotion_allowed": False,
        "safety": {
            "label_source": "verified_audio_training_corpus_only",
            "generated_stems_never_ground_truth": True,
            "self_labeled_ground_truth": False,
            "weak_labels_created": False,
            "tensor_backed_rows_required": True,
            "short_tensor_aliases": True,
        },
        **SAFETY_FLAGS,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    append_progress(progress_path, "finish", start_time=start_time, row_count=len(rows), blocked_row_count=len(blocked), build_status=manifest["build_status"])
    return manifest


def build_audio_quality_verified_rows(
    *,
    verified_corpus_manifest_json: Path,
    output_dir: Path,
    rag_current_json: Path,
    source_lock_pack_json: Path,
    window_sec: float = 6.0,
    tensor_bins: int = 64,
) -> dict[str, Any]:
    if window_sec <= 0:
        raise ValueError("window_sec_must_be_positive")
    if tensor_bins <= 0:
        raise ValueError("tensor_bins_must_be_positive")
    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILE
    if progress_path.exists():
        progress_path.unlink()
    append_progress(
        progress_path,
        "start",
        start_time=start_time,
        verified_corpus_manifest_json=str(verified_corpus_manifest_json),
        rag_current_json=str(rag_current_json),
        source_lock_pack_json=str(source_lock_pack_json),
        expected_acceleration_mode="cpu_audio_tensor_extraction",
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="verified_row_intake_no_model_inference",
    )

    corpus_manifest = read_json(verified_corpus_manifest_json)
    base_dir = verified_corpus_manifest_json.parent
    audio_training_examples_jsonl = _resolve_path(corpus_manifest.get("audio_training_examples_jsonl"), base_dir=base_dir)
    row_report_jsonl = _resolve_path(corpus_manifest.get("row_report_jsonl"), base_dir=base_dir)
    errors: list[str] = []
    if corpus_manifest.get("schema") != VERIFIED_CORPUS_SCHEMA:
        errors.append("unexpected_verified_corpus_manifest_schema")
    errors.extend(_manifest_flag_reasons("verified_corpus_manifest", corpus_manifest))
    source_lock_errors = _source_lock_reasons(source_lock_pack_json, rag_current_json=rag_current_json)

    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    examples = list(iter_jsonl(audio_training_examples_jsonl)) if audio_training_examples_jsonl.exists() else []
    reports = _verified_report_map(row_report_jsonl)

    if corpus_manifest.get("corpus_validation_status") != "passed":
        errors.append("corpus_validation_not_passed")
        errors.extend(source_lock_errors)
        blocked = [_blocked_row(row, ["corpus_validation_not_passed"]) for row in examples]
        return _write_outputs(
            output_dir=output_dir,
            progress_path=progress_path,
            verified_corpus_manifest_json=verified_corpus_manifest_json,
            audio_training_examples_jsonl=audio_training_examples_jsonl,
            row_report_jsonl=row_report_jsonl,
            rag_current_json=rag_current_json,
            source_lock_pack_json=source_lock_pack_json,
            rows=[],
            blocked=blocked or [_blocked_row({}, ["corpus_validation_not_passed"])],
            errors=errors,
            start_time=start_time,
        )

    if not audio_training_examples_jsonl.exists():
        errors.append("audio_training_examples_jsonl_missing")
    if not row_report_jsonl.exists():
        errors.append("row_report_jsonl_missing")
    errors.extend(source_lock_errors)
    rag_refs = _rag_refs(rag_current_json, verified_corpus_manifest_json)
    if not rag_refs:
        errors.append("verified_corpus_rag_ref_missing")
    if errors:
        return _write_outputs(
            output_dir=output_dir,
            progress_path=progress_path,
            verified_corpus_manifest_json=verified_corpus_manifest_json,
            audio_training_examples_jsonl=audio_training_examples_jsonl,
            row_report_jsonl=row_report_jsonl,
            rag_current_json=rag_current_json,
            source_lock_pack_json=source_lock_pack_json,
            rows=[],
            blocked=[_blocked_row({}, errors)],
            errors=errors,
            start_time=start_time,
        )

    for source_index, row in enumerate(examples, start=1):
        critical = _critical_ground_truth_claims(row)
        if critical:
            raise ValueError(",".join(critical))
        report = reports.get(str(row.get("id") or ""))
        reasons = _row_block_reasons(row, report, source_lock_pack_json=source_lock_pack_json)
        if reasons:
            blocked.append(_blocked_row(row, reasons))
            continue
        tensor_path, tensor_error = _write_tensor(
            row=row,
            output_dir=output_dir,
            row_index=len(rows) + 1,
            window_sec=window_sec,
            tensor_bins=tensor_bins,
        )
        if tensor_error:
            blocked.append(_blocked_row(row, [tensor_error]))
            continue
        rows.append(
            _aqc_row(
                row=row,
                tensor_path=tensor_path,
                row_index=source_index,
                verified_corpus_manifest_json=verified_corpus_manifest_json,
                audio_training_examples_jsonl=audio_training_examples_jsonl,
                row_report_jsonl=row_report_jsonl,
                rag_refs=rag_refs,
                source_lock_pack_json=source_lock_pack_json,
                window_sec=window_sec,
            )
        )
        append_progress(progress_path, "row", start_time=start_time, row_count=len(rows), label_type=rows[-1]["label_type"], split=rows[-1]["split"])

    if blocked:
        errors.append("blocked_rows_present")
    return _write_outputs(
        output_dir=output_dir,
        progress_path=progress_path,
        verified_corpus_manifest_json=verified_corpus_manifest_json,
        audio_training_examples_jsonl=audio_training_examples_jsonl,
        row_report_jsonl=row_report_jsonl,
        rag_current_json=rag_current_json,
        source_lock_pack_json=source_lock_pack_json,
        rows=rows,
        blocked=blocked,
        errors=errors,
        start_time=start_time,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tensor-backed AQC verified rows from a validated verified audio corpus.")
    parser.add_argument("--verified-corpus-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-current-json", type=Path, required=True)
    parser.add_argument("--source-lock-pack-json", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--tensor-bins", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_audio_quality_verified_rows.py",
        entrypoint_path="/app/tools/build_audio_quality_verified_rows.py",
    )
    args = parse_args(argv)
    manifest = build_audio_quality_verified_rows(
        verified_corpus_manifest_json=args.verified_corpus_manifest_json,
        output_dir=args.output_dir,
        rag_current_json=args.rag_current_json,
        source_lock_pack_json=args.source_lock_pack_json,
        window_sec=float(args.window_sec),
        tensor_bins=int(args.tensor_bins),
    )
    print(
        json.dumps(
            {
                "manifest_json": str(args.output_dir / MANIFEST_FILE),
                "build_status": manifest["build_status"],
                "row_count": manifest["row_count"],
                "tensor_row_count": manifest["tensor_row_count"],
                "errors": manifest["errors"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["build_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
