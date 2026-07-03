#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_audio_error_dataset_provenance import is_real_value, is_synthetic_candidate_id  # noqa: E402
from tools.build_audio_error_training_views import (  # noqa: E402
    MANIFEST_FILE,
    VIEW_FILES,
    VIEW_READINESS_RULES,
    readiness_blockers,
    trainable_safe_rows,
)


REQUIRED_ROW_FIELDS = {
    "view_name",
    "target_name",
    "row_id",
    "track_id",
    "run_id",
    "candidate_id",
    "route_id",
    "stem_type",
    "window_start_sec",
    "window_end_sec",
    "candidate_audio_path",
    "mix_path",
    "error_map_path",
    "feature_values",
    "label",
    "label_source",
    "blocker_labels",
    "issue_labels",
    "corrective_action_labels",
    "corrective_target_lanes",
    "unmapped_blocker_labels",
    "corrective_status",
    "provenance_source_files",
    "safe_for_this_view",
    "unsafe_reason_for_this_view",
    "release_ready",
}
ACOUSTIC_FEATURE_METADATA_COLUMNS = {"route_id", "candidate_family", "stem_type", "window_start_sec", "window_end_sec", "window_duration_sec"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def scalar_feature_present(row: dict[str, Any]) -> bool:
    features = row.get("feature_values")
    return isinstance(features, dict) and bool(features)


def error_map_present(row: dict[str, Any]) -> bool:
    text = str(row.get("error_map_path") or "").strip()
    return bool(text and Path(text).exists())


def acoustic_feature_present(row: dict[str, Any]) -> bool:
    features = row.get("feature_values")
    if not isinstance(features, dict):
        return False
    for name, value in features.items():
        if name in ACOUSTIC_FEATURE_METADATA_COLUMNS:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if abs(numeric) > 1.0e-12:
            return True
    return False


def validate_window(row: dict[str, Any], row_key: str, errors: list[str]) -> None:
    try:
        if float(row.get("window_end_sec") or 0.0) <= float(row.get("window_start_sec") or 0.0):
            errors.append(f"row_invalid_window_boundaries:{row_key}")
    except (TypeError, ValueError):
        errors.append(f"row_invalid_window_boundaries:{row_key}")


def validate_safe_row(row: dict[str, Any], row_key: str, errors: list[str]) -> None:
    for field in ("track_id", "run_id", "candidate_id", "route_id", "mix_path", "candidate_audio_path"):
        if not is_real_value(row.get(field)):
            errors.append(f"safe_row_missing_{field}:{row_key}")
    for field in ("issue_labels", "corrective_action_labels", "corrective_target_lanes", "unmapped_blocker_labels"):
        if not isinstance(row.get(field), list):
            errors.append(f"safe_row_invalid_{field}:{row_key}")
    if str(row.get("corrective_status") or "") not in {"not_required", "actionable_hardcase"}:
        errors.append(f"safe_row_invalid_corrective_status:{row_key}")
    if str(row.get("route_id") or "").strip().lower() == "unknown":
        errors.append(f"safe_row_unknown_route_id:{row_key}")
    if is_synthetic_candidate_id(row.get("candidate_id")):
        errors.append(f"safe_row_synthetic_candidate_id:{row_key}")
    if row.get("label") not in (0, 1):
        errors.append(f"safe_row_invalid_label:{row_key}")
    if not is_real_value(row.get("label_source")):
        errors.append(f"safe_row_missing_label_source:{row_key}")
    provenance = row.get("provenance_source_files")
    if not isinstance(provenance, list) or not provenance:
        errors.append(f"safe_row_missing_provenance_source_files:{row_key}")
    if row.get("unsafe_reason_for_this_view"):
        errors.append(f"safe_row_has_unsafe_reason:{row_key}")
    if not error_map_present(row) and not scalar_feature_present(row):
        errors.append(f"safe_row_missing_error_map_or_features:{row_key}")
    if not error_map_present(row) and not acoustic_feature_present(row):
        errors.append(f"safe_row_missing_acoustic_feature_values:{row_key}")


def validate_unsafe_row(row: dict[str, Any], row_key: str, errors: list[str]) -> None:
    if row.get("label") is not None:
        errors.append(f"unsafe_row_has_label:{row_key}")
    if not row.get("unsafe_reason_for_this_view"):
        errors.append(f"unsafe_row_missing_reason:{row_key}")


def summarize_view(view_name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    safe_rows = [row for row in rows if row.get("safe_for_this_view") is True]
    unsafe_rows = [row for row in rows if row.get("safe_for_this_view") is not True]
    training_rows = trainable_safe_rows(view_name, safe_rows)
    positives = sum(1 for row in safe_rows if row.get("label") == 1)
    negatives = sum(1 for row in safe_rows if row.get("label") == 0)
    training_positives = sum(1 for row in training_rows if row.get("label") == 1)
    training_negatives = sum(1 for row in training_rows if row.get("label") == 0)
    reasons = Counter(
        reason
        for row in unsafe_rows
        for reason in str(row.get("unsafe_reason_for_this_view") or "").split(";")
        if reason
    )
    blockers = readiness_blockers(view_name, training_rows)
    return {
        "rows_total": len(rows),
        "safe_rows": len(safe_rows),
        "unsafe_rows": len(unsafe_rows),
        "training_safe_rows": len(training_rows),
        "score_only_safe_rows": len(safe_rows) - len(training_rows),
        "positive_labels": positives,
        "negative_labels": negatives,
        "training_positive_labels": training_positives,
        "training_negative_labels": training_negatives,
        "unsafe_reason_counts": dict(sorted(reasons.items())),
        "readiness_blockers": blockers,
        "can_train": not blockers,
    }


def validate_audio_error_training_views(views_dir: Path) -> dict[str, Any]:
    manifest_path = views_dir / MANIFEST_FILE
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("missing_training_views_manifest")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_must_be_false:manifest")
    if manifest.get("training_started") is True or manifest.get("separator_training_started") is True:
        errors.append("training_must_not_start:manifest")
    if manifest.get("melody_transcription_ran") is True:
        errors.append("melody_transcription_must_not_run:manifest")
    if manifest.get("recommendation_index_updated") is True:
        errors.append("recommendation_index_must_not_update:manifest")

    view_results: dict[str, dict[str, Any]] = {}
    rows_checked = 0
    for view_name, filename in VIEW_FILES.items():
        path = views_dir / filename
        if not path.exists():
            errors.append(f"missing_view_file:{filename}")
            rows: list[dict[str, Any]] = []
        else:
            rows = list(iter_jsonl(path))
        for index, row in enumerate(rows):
            rows_checked += 1
            row_key = f"{view_name}:{index}:{row.get('row_id', '')}"
            missing = sorted(REQUIRED_ROW_FIELDS - set(row))
            if missing:
                errors.append(f"row_missing_fields:{row_key}:{','.join(missing)}")
            if row.get("release_ready") is True:
                errors.append(f"release_ready_must_be_false:row:{row_key}")
            if row.get("view_name") != view_name:
                errors.append(f"row_view_name_mismatch:{row_key}:{row.get('view_name')}")
            validate_window(row, row_key, errors)
            if row.get("safe_for_this_view") is True:
                validate_safe_row(row, row_key, errors)
            else:
                validate_unsafe_row(row, row_key, errors)
        view_results[view_name] = summarize_view(view_name, rows)

    result = {
        "ok": not errors,
        "errors": errors,
        "rows_checked": rows_checked,
        "views": view_results,
        "readiness_minimums": VIEW_READINESS_RULES,
        "can_train_minus_critic": view_results.get("minus_critic", {}).get("can_train") is True,
        "can_train_vocal_critic": view_results.get("vocal_critic", {}).get("can_train") is True,
        "can_train_pair_critic": view_results.get("pair_critic", {}).get("can_train") is True,
        "can_train_overall_accept": view_results.get("overall_accept", {}).get("can_train") is True,
        "release_ready": False,
        "training_started": False,
        "separator_training_started": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate target-specific audio error critic training views.")
    parser.add_argument("--views-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_audio_error_training_views(args.views_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
