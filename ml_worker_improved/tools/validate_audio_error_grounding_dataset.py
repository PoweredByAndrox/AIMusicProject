#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.extract_audio_error_maps import ERROR_MAP_NAMES


REQUIRED_ROW_FIELDS = {
    "dataset_id",
    "track_id",
    "run_id",
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "mix_path",
    "vocal_candidate_path",
    "minus_candidate_path",
    "candidate_id",
    "route_id",
    "stem_type",
    "blockers",
    "judge_decision",
    "pair_compatibility_status",
    "residual_energy",
    "mixture_consistency_error",
    "phase_collapse_score",
    "stereo_width_score",
    "vocal_bleed_proxy_score",
    "instrumental_bleed_proxy_score",
    "missing_detail_score",
    "transient_damage_score",
    "artifact_noise_score",
    "error_map_paths",
    "train_split",
    "safe_for_training",
    "unsafe_reason",
}


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


def has_true_release_flag(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "release_ready" and value is True:
                return True
            if key in {"melody_transcription_ran", "recommendation_index_updated"} and value is True:
                return True
            if has_true_release_flag(value):
                return True
    if isinstance(payload, list):
        return any(has_true_release_flag(item) for item in payload)
    return False


def validate_npz(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing_error_map_npz:{path}"]
    try:
        arrays = np.load(path)
    except Exception as exc:  # pragma: no cover - corrupt external file path.
        return [f"unreadable_error_map_npz:{path}:{type(exc).__name__}:{exc}"]
    for name in ERROR_MAP_NAMES:
        if name not in arrays.files:
            errors.append(f"missing_map_array:{path}:{name}")
            continue
        data = arrays[name]
        if data.size == 0:
            errors.append(f"empty_map_array:{path}:{name}")
        elif not np.isfinite(data).all():
            errors.append(f"non_finite_map_array:{path}:{name}")
    return errors


def validate_audio_error_grounding_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "audio_error_grounding_manifest.json"
    rows_path = dataset_dir / "audio_error_window_dataset.jsonl"
    maps_index_path = dataset_dir / "audio_error_maps_index.jsonl"
    manifest = read_json(manifest_path)
    rows = list(iter_jsonl(rows_path))
    maps_index = list(iter_jsonl(maps_index_path))
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("missing_manifest")
    if not rows_path.exists():
        errors.append("missing_window_dataset")
    if not maps_index_path.exists():
        errors.append("missing_maps_index")
    if manifest.get("release_ready") is True:
        errors.append("release_ready_must_be_false:manifest")
    if manifest.get("melody_transcription_ran") is True:
        errors.append("melody_transcription_must_not_run:manifest")
    if manifest.get("recommendation_index_updated") is True:
        errors.append("recommendation_index_must_not_update:manifest")
    if has_true_release_flag(rows):
        errors.append("release_ready_must_be_false:rows")
    if has_true_release_flag(maps_index):
        errors.append("release_ready_must_be_false:maps_index")

    safe_for_training_count = 0
    unsafe_count = 0
    for index, row in enumerate(rows):
        missing_fields = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing_fields:
            errors.append(f"row_missing_fields:{index}:{','.join(missing_fields)}")
        for field in ("candidate_id", "route_id", "judge_decision", "pair_compatibility_status"):
            if str(row.get(field) or "") == "unknown":
                errors.append(f"row_forbidden_unknown_value:{index}:{field}")
        for field in ("mix_path", "vocal_candidate_path", "minus_candidate_path"):
            if not row.get(field):
                errors.append(f"row_missing_audio_path:{index}:{field}")
        if float(row.get("window_end_sec") or 0.0) <= float(row.get("window_start_sec") or 0.0):
            errors.append(f"row_invalid_window_boundaries:{index}:{row.get('window_id')}")
        safe = bool(row.get("safe_for_training"))
        if safe:
            safe_for_training_count += 1
            for field in ("track_id", "run_id", "candidate_id", "route_id", "judge_decision", "pair_compatibility_status"):
                if not row.get(field):
                    errors.append(f"safe_row_missing_provenance:{index}:{field}")
            if row.get("unsafe_reason"):
                errors.append(f"safe_row_has_unsafe_reason:{index}:{row.get('window_id')}")
            paths = row.get("error_map_paths")
            if not isinstance(paths, dict) or not paths.get("npz"):
                errors.append(f"safe_row_missing_error_map:{index}:{row.get('window_id')}")
            else:
                errors.extend(validate_npz(Path(paths["npz"])))
        else:
            unsafe_count += 1
            if not row.get("unsafe_reason"):
                errors.append(f"unsafe_row_missing_reason:{index}:{row.get('window_id')}")
    result = {
        "ok": not errors,
        "errors": errors,
        "rows_checked": len(rows),
        "maps_index_rows": len(maps_index),
        "safe_for_training_count": safe_for_training_count,
        "unsafe_count": unsafe_count,
        "release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a no-reference audio error grounding dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_audio_error_grounding_dataset(args.dataset_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
