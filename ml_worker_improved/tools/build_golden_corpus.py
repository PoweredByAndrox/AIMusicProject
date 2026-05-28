from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from tools.artifact_status_contract import ArtifactContractError, stable_track_hash, status_view_from_manifest


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


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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


def build_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in iter_manifest_paths(input_root):
        payload = load_json(manifest_path)
        if payload is None:
            continue
        view = status_view_from_manifest(manifest_path)
        try:
            view.assert_promotable()
        except ArtifactContractError:
            continue
        track_hash = manifest_track_hash(payload, manifest_path)
        rows.append(
            {
                "case_id": f"golden_{track_hash}_{len(rows):05d}",
                "track_hash": track_hash,
                "split": "holdout",
                "source_manifest": view.source_manifest,
                "manifest_sha256": view.manifest_sha256,
                "stem_type": view.stem_type,
                "delivery_status": view.delivery_status,
                "delivery_allowed": view.delivery_allowed,
                "presentation_status": view.presentation_status,
                "release_status": view.release_status,
                "training_allowed": False,
                "threshold_tuning_allowed": False,
                "golden_role": "protected_regression_holdout",
                "audio_path": view.audio_path,
                "model_id": view.model_id,
                "candidate_id": view.candidate_id,
            }
        )
    return sorted(rows, key=lambda row: (row["track_hash"], row["source_manifest"]))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_extract_zip(input_zip: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    root = work_dir.resolve()
    with zipfile.ZipFile(input_zip) as archive:
        for member in archive.infolist():
            target = (work_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe_zip_member:{member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    return work_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--input-zip", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.input_zip:
        input_root = safe_extract_zip(args.input_zip, args.output_jsonl.parent / "_golden_zip_extract")
    elif args.input_root:
        input_root = args.input_root
    else:
        parser.error("one of --input-root or --input-zip is required")

    rows = build_rows(input_root)
    write_jsonl(args.output_jsonl, rows)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(
            {
                "schema": "golden_corpus_manifest.v1",
                "input_root": str(input_root),
                "row_count": len(rows),
                "training_allowed": False,
                "threshold_tuning_allowed": False,
                "role": "protected_regression_holdout",
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
