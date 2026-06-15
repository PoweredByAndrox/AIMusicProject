from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.extract_audio_error_maps import extract_error_maps_for_window


SAFETY_FLAGS = {
    "training_started": False,
    "critic_training_started": False,
    "separator_training_started": False,
    "repair_adapter_training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "release_ready": False,
    "can_publish_as_result": False,
}
START_TIME = time.monotonic()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected_json_object:{path}:{line_number}")
        yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, *, stage: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "stage": stage,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "cpu_tensor_map_extraction",
        "gpu_inference_skipped": True,
        "gpu_inference_skipped_reason": "tensor_dataset_build_uses_audio_decode_and_stft_map_extraction_only",
        "release_ready": False,
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def split_pair_key(row: dict[str, Any]) -> tuple[str, str, str, float, float]:
    return (
        str(row.get("track_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("split") or ""),
        round(as_float(row.get("window_start_sec")), 3),
        round(as_float(row.get("window_end_sec")), 3),
    )


def validate_locked_split(split_dir: Path) -> dict[str, Any]:
    manifest_path = split_dir / "leak_safe_split_manifest.json"
    leakage_path = split_dir / "leakage_report.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing_leak_safe_split_manifest:{manifest_path}")
    if not leakage_path.exists():
        raise FileNotFoundError(f"missing_leakage_report:{leakage_path}")
    manifest = read_json(manifest_path)
    leakage = read_json(leakage_path)
    critical = int(manifest.get("critical_leakage_count") or leakage.get("critical_leakage_count") or 0)
    if critical != 0:
        raise ValueError("critical_leakage_present")
    if manifest.get("release_ready") is True or leakage.get("release_ready") is True:
        raise ValueError("release_ready_split_not_allowed")
    return manifest


def save_map_npz(path: Path, maps: dict[str, Any]) -> list[str]:
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in maps.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sorted(arrays)


def collect_complete_pairs(rows: Sequence[dict[str, Any]]) -> list[tuple[tuple[str, str, str, float, float], dict[str, dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str, float, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[split_pair_key(row)][str(row.get("stem_type") or "")] = row
    complete = [(key, stems) for key, stems in grouped.items() if "vocal" in stems and "minus" in stems]
    return sorted(complete, key=lambda item: stable_hash(list(item[0])))


def build_locked_tensor_audio_error_dataset(
    *,
    split_dir: Path,
    output_dir: Path,
    max_pairs: int = 0,
) -> dict[str, Any]:
    split_manifest = validate_locked_split(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "tensor_dataset_progress.jsonl"
    progress(progress_path, "start", stage="build_locked_tensor_audio_error_dataset", split_dir=str(split_dir))
    split_rows = list(iter_jsonl(split_dir / "leak_safe_split_rows.jsonl"))
    pairs = collect_complete_pairs(split_rows)
    if max_pairs > 0:
        pairs = pairs[: int(max_pairs)]
    tensor_rows: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []
    maps_dir = output_dir / "tensor_maps"
    for index, (key, stems) in enumerate(pairs, start=1):
        vocal = stems["vocal"]
        minus = stems["minus"]
        map_id = stable_hash({"key": key, "vocal": vocal.get("row_uid"), "minus": minus.get("row_uid")})[:24]
        map_path = maps_dir / f"{map_id}.npz"
        try:
            extracted = extract_error_maps_for_window(
                mix_path=Path(str(vocal.get("mix_path") or minus.get("mix_path") or "")),
                vocal_path=Path(str(vocal.get("candidate_audio_path") or "")),
                minus_path=Path(str(minus.get("candidate_audio_path") or "")),
                window_start_sec=as_float(vocal.get("window_start_sec")),
                window_end_sec=as_float(vocal.get("window_end_sec")),
            )
            map_names = save_map_npz(map_path, extracted["maps"])
        except Exception as exc:  # noqa: BLE001
            skipped_pairs.append(
                {
                    "pair_key": list(key),
                    "reason": f"{type(exc).__name__}:{exc}",
                    "release_ready": False,
                }
            )
            continue
        for stem_type, row in (("vocal", vocal), ("minus", minus)):
            tensor_rows.append(
                {
                    "row_uid": str(row.get("row_uid") or ""),
                    "canonical_row_key": str(row.get("canonical_row_key") or ""),
                    "track_id": str(row.get("track_id") or ""),
                    "run_id": str(row.get("run_id") or ""),
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "route_id": str(row.get("route_id") or ""),
                    "candidate_family": str(row.get("candidate_family") or ""),
                    "source_model_family": str(row.get("source_model_family") or ""),
                    "stem_type": stem_type,
                    "split": str(row.get("split") or ""),
                    "label": int(row.get("label") or 0),
                    "train_allowed": row.get("train_allowed") is True,
                    "hardcase": row.get("hardcase") is True,
                    "hardcase_holdout": row.get("hardcase_holdout") is True,
                    "blockers": [str(item) for item in row.get("blockers", []) or []],
                    "window_start_sec": as_float(row.get("window_start_sec")),
                    "window_end_sec": as_float(row.get("window_end_sec")),
                    "tensor_map_path": str(map_path),
                    "tensor_map_names": map_names,
                    "tensor_map_source": "locked_split_pair_extraction",
                    "scalar_features_used": False,
                    "release_ready": False,
                }
            )
        if index % 25 == 0:
            progress(progress_path, "progress", stage="extract_tensor_maps", pairs_done=index, rows_written=len(tensor_rows))
    write_jsonl(output_dir / "tensor_audio_error_dataset.jsonl", tensor_rows)
    write_jsonl(output_dir / "tensor_audio_error_skipped_pairs.jsonl", skipped_pairs)
    split_counts = dict(sorted(Counter(str(row.get("split") or "") for row in tensor_rows).items()))
    label_counts = dict(sorted(Counter(str(row.get("label")) for row in tensor_rows).items()))
    manifest = {
        "schema": "locked_tensor_audio_error_dataset.v1",
        "source_split_dir": str(split_dir),
        "source_split_manifest": str(split_dir / "leak_safe_split_manifest.json"),
        "source_split_dataset_id": str(split_manifest.get("source_dataset_id") or ""),
        "output_dir": str(output_dir),
        "rows_total": len(tensor_rows),
        "pairs_considered": len(pairs),
        "pairs_extracted": len(pairs) - len(skipped_pairs),
        "pairs_skipped": len(skipped_pairs),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "tensor_maps_required": True,
        "scalar_fallback_used": False,
        "critical_leakage_count": 0,
        "dataset_rows_path": str(output_dir / "tensor_audio_error_dataset.jsonl"),
        "skipped_pairs_path": str(output_dir / "tensor_audio_error_skipped_pairs.jsonl"),
        "progress_path": str(progress_path),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "locked_tensor_audio_error_dataset_manifest.json", manifest)
    progress(progress_path, "finish", stage="build_locked_tensor_audio_error_dataset", rows=len(tensor_rows), skipped_pairs=len(skipped_pairs))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tensor-only error-map dataset from EVAL-1 locked split.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_locked_tensor_audio_error_dataset(split_dir=args.split_dir, output_dir=args.output_dir, max_pairs=args.max_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
