from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tools.artifact_status_contract import (
    BLOCKED_STATUSES,
    ArtifactContractError,
    collect_failure_labels,
    infer_dominant_failure_reason,
    stable_track_hash,
    stable_track_split,
    status_view_from_manifest,
)


MANIFEST_NAMES = {
    "adaptive_manifest.json",
    "adaptive_vocal_manifest.json",
    "model_zoo_benchmark_manifest.json",
    "selected_delivery_gate_manifest.json",
    "stem_manifest.json",
    "validation_summary.json",
}
MAX_JSON_BYTES = 5 * 1024 * 1024


def iter_manifest_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.stat().st_size > MAX_JSON_BYTES:
            continue
        if path.name in MANIFEST_NAMES or "manifest" in path.name or path.name == "validation_summary.json":
            yield path


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def extract_windows(payload: Any) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in {"failed_windows", "hard_windows", "issue_windows", "segments", "windows"} and isinstance(value, list):
                windows.extend(item for item in value if isinstance(item, dict))
            else:
                windows.extend(extract_windows(value))
    elif isinstance(payload, list):
        for item in payload:
            windows.extend(extract_windows(item))
    return windows


def window_bounds(window: dict[str, Any]) -> tuple[Any, Any]:
    start = window.get("window_start_sec", window.get("start_sec", window.get("start")))
    end = window.get("window_end_sec", window.get("end_sec", window.get("end")))
    return start, end


def manifest_track_hash(payload: dict[str, Any], manifest_path: Path) -> str:
    existing = str(payload.get("track_hash") or "").strip()
    if existing:
        return existing
    return stable_track_hash(
        payload.get("source_audio_sha256"),
        payload.get("track_id"),
        payload.get("audio_path"),
        manifest_path.parent,
    )


def hard_case_labels(payload: dict[str, Any], dominant: str) -> list[str]:
    labels = sorted(collect_failure_labels(payload))
    if labels:
        return labels
    if dominant and dominant not in {"", "unknown", "none"}:
        return [dominant]
    return ["unknown_blocked_reason"]


def build_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in iter_manifest_paths(input_root):
        payload = read_json(manifest_path)
        if payload is None:
            continue
        view = status_view_from_manifest(manifest_path)
        try:
            view.assert_promotable()
            continue
        except ArtifactContractError:
            pass
        values = {view.delivery_status, view.presentation_status, view.release_status}
        if not view.blocked and not (values & BLOCKED_STATUSES):
            continue

        dominant = view.dominant_failure_reason or infer_dominant_failure_reason(payload) or "unknown_blocked_reason"
        if dominant in {"", "unknown", "none"}:
            dominant = "unknown_blocked_reason"
        labels = hard_case_labels(payload, dominant)
        track_hash = manifest_track_hash(payload, manifest_path)
        windows = extract_windows(payload) or [{}]
        for window in windows:
            start, end = window_bounds(window)
            rows.append(
                {
                    "hard_case_id": f"hard_{track_hash}_{len(rows):05d}",
                    "track_hash": track_hash,
                    "split": stable_track_split(track_hash),
                    "stem_type": view.stem_type,
                    "source_manifest": view.source_manifest,
                    "manifest_sha256": view.manifest_sha256,
                    "delivery_status": view.delivery_status,
                    "delivery_allowed": view.delivery_allowed,
                    "presentation_status": view.presentation_status,
                    "release_status": view.release_status,
                    "dominant_failure_reason": dominant,
                    "failure_labels": labels,
                    "window_start_sec": start,
                    "window_end_sec": end,
                    "audio_path": view.audio_path,
                    "model_id": view.model_id,
                    "candidate_id": view.candidate_id,
                }
            )
    return sorted(rows, key=lambda row: (row["track_hash"], row["source_manifest"], str(row["window_start_sec"])))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.input_root)
    write_jsonl(args.output_jsonl, rows)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(
            {
                "schema": "hard_case_corpus_manifest.v1",
                "input_root": str(args.input_root),
                "row_count": len(rows),
                "split_rule": "stable_track_split(track_hash)",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
