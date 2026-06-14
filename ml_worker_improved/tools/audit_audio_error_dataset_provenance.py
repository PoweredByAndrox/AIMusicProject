#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "release_ready": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
}

DATASET_ROWS = "audio_error_window_dataset.jsonl"
DATASET_MANIFEST = "audio_error_grounding_manifest.json"
AUDIT_JSON = "audio_error_provenance_audit.json"
AUDIT_JSONL = "audio_error_provenance_audit.jsonl"
AUDIT_SUMMARY = "audio_error_provenance_audit_summary.md"

CANDIDATE_FILES = (
    "candidate_pool.jsonl",
    "selected_candidates.jsonl",
    "candidate_wave_jobs.jsonl",
    "candidate_job_plan.jsonl",
    "rendered_preview_audio.jsonl",
)
PAIR_FILES = ("pair_compatibility.jsonl",)
JUDGE_FILES = (
    "minus_windowed_judge.jsonl",
    "minus_judge.jsonl",
    "vocal_windowed_judge.jsonl",
    "vocal_judge.jsonl",
)
JSON_FILES = ("run_manifest.json", "preview_manifest.json")

CANDIDATE_ID_KEYS = (
    "candidate_id",
    "selected_candidate_id",
    "selected_vocal_candidate_id",
    "selected_minus_candidate_id",
    "source_candidate_id",
    "input_candidate_id",
)
ROUTE_KEYS = ("route_id",)
ROUTE_FALLBACK_KEYS = ("candidate_config_id",)
PAIR_STATUS_KEYS = ("pair_compatibility_status", "pair_decision", "compatibility_status")
PATH_KEYS = (
    "mix_path",
    "input_audio_path",
    "input_audio",
    "source_mix_path",
    "vocal_path",
    "vocal_candidate_path",
    "selected_vocal_path",
    "normalized_vocal_path",
    "raw_vocal_path",
    "minus_path",
    "instrumental_path",
    "minus_candidate_path",
    "selected_minus_path",
    "normalized_minus_path",
    "normalized_instrumental_path",
    "raw_minus_path",
)
DATASET_STEM_PATH_KEYS = ("vocal_candidate_path", "minus_candidate_path")
SYNTHETIC_TOKENS = ("synthetic", "unknown", "placeholder", "fake", "missing")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_path(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("\\", "/")


def is_real_value(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return lowered != "unknown" and not lowered.startswith("unknown_")


def is_synthetic_candidate_id(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(text == token or text.startswith(f"{token}_") for token in SYNTHETIC_TOKENS)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def collect_candidate_ids(row: dict[str, Any]) -> set[str]:
    return {str(row.get(key)) for key in CANDIDATE_ID_KEYS if is_real_value(row.get(key))}


def collect_paths(payload: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in PATH_KEYS and isinstance(value, str) and value:
                paths.add(normalize_path(value))
            elif key in {"output_paths", "audio_shape", "shape_normalization_outputs"}:
                paths.update(collect_paths(value))
            elif isinstance(value, dict):
                paths.update(collect_paths(value))
    return paths


def route_value(row: dict[str, Any]) -> tuple[str, str]:
    for key in ROUTE_KEYS:
        if is_real_value(row.get(key)):
            return str(row[key]), key
    for key in ROUTE_FALLBACK_KEYS:
        if is_real_value(row.get(key)):
            return str(row[key]), f"{key}_as_route_id"
    return "", ""


def pair_status_value(row: dict[str, Any]) -> str:
    for key in PAIR_STATUS_KEYS:
        if is_real_value(row.get(key)):
            return str(row[key])
    return ""


def dataset_row_id(index: int, row: dict[str, Any]) -> str:
    return str(row.get("row_id") or index)


def source_file_label(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def track_matches(dataset_row: dict[str, Any], source_row: dict[str, Any]) -> bool:
    dataset_track = str(dataset_row.get("track_id") or "")
    source_track = str(source_row.get("track_id") or "")
    return not dataset_track or not source_track or dataset_track == source_track


class SourceIndex:
    def __init__(self) -> None:
        self.candidate_records: list[dict[str, Any]] = []
        self.pair_records: list[dict[str, Any]] = []
        self.judge_records: list[dict[str, Any]] = []
        self.files_used: set[str] = set()

    def add_candidate_record(self, *, row: dict[str, Any], source_file: str, artifact_type: str) -> None:
        route_id, route_source_field = route_value(row)
        record = {
            "artifact_type": artifact_type,
            "source_file": source_file,
            "row": row,
            "candidate_ids": collect_candidate_ids(row),
            "paths": collect_paths(row),
            "route_id": route_id,
            "route_source_field": route_source_field,
        }
        self.candidate_records.append(record)
        self.files_used.add(source_file)

    def add_pair_record(self, *, row: dict[str, Any], source_file: str) -> None:
        record = {
            "artifact_type": "pair_compatibility",
            "source_file": source_file,
            "row": row,
            "candidate_ids": collect_candidate_ids(row),
            "pair_compatibility_status": pair_status_value(row),
        }
        self.pair_records.append(record)
        self.files_used.add(source_file)

    def add_judge_record(
        self,
        *,
        row: dict[str, Any],
        window: dict[str, Any],
        source_file: str,
        stem_type: str,
    ) -> None:
        start = float(window.get("start_sec") or window.get("window_start_sec") or 0.0)
        duration = float(window.get("duration_sec") or 0.0)
        end = float(window.get("end_sec") or window.get("window_end_sec") or (start + duration))
        record = {
            "artifact_type": "window_judge",
            "source_file": source_file,
            "row": row,
            "window": window,
            "stem_type": stem_type,
            "candidate_ids": collect_candidate_ids(row),
            "start_sec": round(start, 6),
            "end_sec": round(end, 6),
        }
        self.judge_records.append(record)
        self.files_used.add(source_file)


def load_source_index(run_roots: Sequence[Path]) -> SourceIndex:
    index = SourceIndex()
    for run_root in run_roots:
        if not run_root.exists():
            continue
        run_dirs = [run_root] if run_root.is_dir() else [run_root.parent]
        for marker in ("candidate_pool.jsonl", "minus_windowed_judge.jsonl", "pair_compatibility.jsonl"):
            for path in run_root.rglob(marker) if run_root.is_dir() else []:
                if path.parent not in run_dirs:
                    run_dirs.append(path.parent)
        for run_dir in sorted(set(run_dirs), key=lambda path: str(path)):
            for filename in CANDIDATE_FILES:
                path = run_dir / filename
                if not path.exists():
                    continue
                artifact_type = filename.removesuffix(".jsonl")
                for row in iter_jsonl(path):
                    index.add_candidate_record(row=row, source_file=source_file_label(run_dir, path), artifact_type=artifact_type)
            for filename in PAIR_FILES:
                path = run_dir / filename
                if path.exists():
                    for row in iter_jsonl(path):
                        index.add_pair_record(row=row, source_file=source_file_label(run_dir, path))
            for filename in JUDGE_FILES:
                path = run_dir / filename
                if not path.exists():
                    continue
                stem_type = "minus" if "minus" in filename else "vocal"
                for row in iter_jsonl(path):
                    windows = row.get("window_metrics")
                    if isinstance(windows, list):
                        for window in windows:
                            if isinstance(window, dict):
                                index.add_judge_record(row=row, window=window, source_file=source_file_label(run_dir, path), stem_type=stem_type)
            for filename in JSON_FILES:
                path = run_dir / filename
                payload = read_json(path)
                if payload:
                    index.add_candidate_record(row=payload, source_file=source_file_label(run_dir, path), artifact_type=filename.removesuffix(".json"))
            for manifest_path in run_dir.rglob("job_manifest.json"):
                payload = read_json(manifest_path)
                if payload:
                    index.add_candidate_record(
                        row=payload,
                        source_file=source_file_label(run_dir, manifest_path),
                        artifact_type="job_manifest",
                    )
    return index


def row_stem_paths(row: dict[str, Any]) -> set[str]:
    return {normalize_path(row.get(key)) for key in DATASET_STEM_PATH_KEYS if normalize_path(row.get(key))}


def record_matches_row(record: dict[str, Any], row: dict[str, Any]) -> bool:
    source_row = record["row"]
    if not track_matches(row, source_row):
        return False
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id and candidate_id in record["candidate_ids"]:
        return True
    stem_paths = row_stem_paths(row)
    return bool(stem_paths and stem_paths.intersection(record["paths"]))


def matching_candidate_records(index: SourceIndex, row: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in index.candidate_records if record_matches_row(record, row)]


def matching_pair_records(index: SourceIndex, row: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        return []
    result = []
    for record in index.pair_records:
        if candidate_id not in record["candidate_ids"]:
            continue
        if track_matches(row, record["row"]):
            result.append(record)
    return result


def matching_window_records(index: SourceIndex, row: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        return []
    start = round(float(row.get("window_start_sec") or 0.0), 6)
    end = round(float(row.get("window_end_sec") or 0.0), 6)
    stem_type = str(row.get("stem_type") or "")
    matches = []
    for record in index.judge_records:
        if record["stem_type"] != stem_type:
            continue
        if candidate_id not in record["candidate_ids"]:
            continue
        if not track_matches(row, record["row"]):
            continue
        if record["start_sec"] == start and record["end_sec"] == end:
            matches.append(record)
    return matches


def first_real_route(records: list[dict[str, Any]]) -> tuple[str, str, str]:
    for record in records:
        route_id = str(record.get("route_id") or "")
        if is_real_value(route_id):
            return route_id, str(record.get("route_source_field") or ""), str(record["source_file"])
    return "", "", ""


def first_real_candidate_id(records: list[dict[str, Any]]) -> tuple[str, str]:
    candidates = sorted({candidate_id for record in records for candidate_id in record["candidate_ids"] if is_real_value(candidate_id)})
    if len(candidates) == 1:
        source_file = next(record["source_file"] for record in records if candidates[0] in record["candidate_ids"])
        return candidates[0], str(source_file)
    return "", ""


def first_pair_status(records: list[dict[str, Any]]) -> tuple[str, str]:
    for record in records:
        status = str(record.get("pair_compatibility_status") or "")
        if is_real_value(status):
            return status, str(record["source_file"])
    return "", ""


def missing_provenance_fields(row: dict[str, Any]) -> list[str]:
    missing = []
    if not is_real_value(row.get("candidate_id")):
        missing.append("candidate_id")
    if not is_real_value(row.get("route_id")):
        missing.append("route_id")
    if not is_real_value(row.get("pair_compatibility_status")):
        missing.append("pair_compatibility_status")
    return missing


def audit_row(index: SourceIndex, row: dict[str, Any], row_index: int) -> dict[str, Any]:
    missing_fields = missing_provenance_fields(row)
    candidate_records = matching_candidate_records(index, row)
    pair_records = matching_pair_records(index, row)
    window_records = matching_window_records(index, row)
    candidate_pool_records = [record for record in candidate_records if record["artifact_type"] == "candidate_pool"]
    candidate_id, candidate_source = first_real_candidate_id(candidate_records)
    route_id, route_source_field, route_source = first_real_route(candidate_records)
    pair_status, pair_source = first_pair_status(pair_records)
    repair_values = {
        "candidate_id": candidate_id if "candidate_id" in missing_fields else "",
        "route_id": route_id if "route_id" in missing_fields else "",
        "pair_compatibility_status": pair_status if "pair_compatibility_status" in missing_fields else "",
    }
    repair_source_files = stable_unique(
        [
            candidate_source if repair_values["candidate_id"] else "",
            route_source if repair_values["route_id"] else "",
            pair_source if repair_values["pair_compatibility_status"] else "",
            *(str(record["source_file"]) for record in window_records),
        ]
    )
    missing_after = []
    if "candidate_id" in missing_fields and not repair_values["candidate_id"]:
        missing_after.append("missing_candidate_id")
    if "route_id" in missing_fields and not repair_values["route_id"]:
        missing_after.append("missing_route_id")
    if "pair_compatibility_status" in missing_fields and not repair_values["pair_compatibility_status"]:
        missing_after.append("missing_pair_compatibility_status")
    if missing_fields and not window_records:
        missing_after.append("missing_window_judge_match")
    if not missing_fields:
        missing_after.append("no_missing_provenance_fields")
    repairable = bool(missing_fields) and not missing_after
    return {
        "row_id": dataset_row_id(row_index, row),
        "row_index": row_index,
        "track_id": str(row.get("track_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "missing_fields": missing_fields,
        "candidate_pool_match_found": bool(candidate_pool_records),
        "route_id_recoverable": bool(repair_values["route_id"]),
        "route_id_repair_source_field": route_source_field if repair_values["route_id"] else "",
        "pair_compatibility_recoverable": bool(repair_values["pair_compatibility_status"]),
        "window_judge_match_found": bool(window_records),
        "repairable": repairable,
        "unrepairable_reason": ";".join(stable_unique(missing_after)),
        "repair_values": repair_values,
        "repair_source_files": repair_source_files,
        "repair_confidence": 1.0 if repairable else 0.0,
        **SAFETY_FLAGS,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Audio Error Dataset Provenance Audit",
        "",
        f"- rows_total: {summary.get('rows_total', 0)}",
        f"- rows_audited: {summary.get('rows_audited', 0)}",
        f"- rows_repairable: {summary.get('rows_repairable', 0)}",
        f"- rows_unrepairable: {summary.get('rows_unrepairable', 0)}",
        f"- missing_route_id: {summary.get('missing_route_id', 0)}",
        f"- missing_pair_compatibility_status: {summary.get('missing_pair_compatibility_status', 0)}",
        f"- route_id_recoverable: {summary.get('route_id_recoverable', 0)}",
        f"- pair_compatibility_recoverable: {summary.get('pair_compatibility_recoverable', 0)}",
        "- release_ready: false",
        "- training_started: false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_audio_error_dataset_provenance(
    *,
    dataset_dir: Path,
    run_roots: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(dataset_dir / DATASET_MANIFEST)
    rows = list(iter_jsonl(dataset_dir / DATASET_ROWS))
    index = load_source_index(run_roots)
    unsafe_rows = [(idx, row) for idx, row in enumerate(rows) if row.get("safe_for_training") is not True]
    audit_rows = [audit_row(index, row, idx) for idx, row in unsafe_rows]
    summary = {
        "dataset_dir": str(dataset_dir),
        "dataset_id": str(manifest.get("dataset_id") or ""),
        "rows_total": len(rows),
        "rows_audited": len(audit_rows),
        "safe_for_training_count_before": sum(1 for row in rows if row.get("safe_for_training") is True),
        "unsafe_count_before": sum(1 for row in rows if row.get("safe_for_training") is not True),
        "missing_route_id": sum(1 for row in rows if not is_real_value(row.get("route_id"))),
        "missing_pair_compatibility_status": sum(1 for row in rows if not is_real_value(row.get("pair_compatibility_status"))),
        "rows_repairable": sum(1 for row in audit_rows if row["repairable"] is True),
        "rows_unrepairable": sum(1 for row in audit_rows if row["repairable"] is not True),
        "route_id_recoverable": sum(1 for row in audit_rows if row["route_id_recoverable"] is True),
        "pair_compatibility_recoverable": sum(1 for row in audit_rows if row["pair_compatibility_recoverable"] is True),
        "source_files_indexed": sorted(index.files_used),
        **SAFETY_FLAGS,
    }
    write_json(output_dir / AUDIT_JSON, summary)
    write_jsonl(output_dir / AUDIT_JSONL, audit_rows)
    write_summary(output_dir / AUDIT_SUMMARY, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit recoverable provenance for audio error grounding dataset rows.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_audio_error_dataset_provenance(
        dataset_dir=args.dataset_dir,
        run_roots=args.run_roots,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
