#!/usr/bin/env python3
"""Triage raw audio folders that might become full-pack reference bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_full_pack_reference_corpus_intake import (  # noqa: E402
    load_gap_rows,
    normalized_stem,
    required_stems_from_gaps,
    stable_unique,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "raw.json"
ROWS_NAME = "rows.jsonl"
REJECTED_NAME = "reject.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
PACK_MANIFEST_NAME = "pack.json"
AGGREGATE_PACK_SCHEMA = "aggregate_short_listen_pack_manifest.v1"
STEM_PACK_SCHEMA = "stem_manifest_audio_pack_manifest.v1"
STEM_TOKENS = {
    "bass": ("bass",),
    "drums": ("drum", "drums", "beat", "beats"),
    "guitar": ("guitar", "guitars"),
    "minus": ("instrumental", "instrumentals", "minus", "karaoke"),
    "other": ("other",),
    "piano": ("piano", "keys", "keyboard"),
    "vocal": ("acapella", "a cappella", "vocal", "vocals", "voice"),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_raw_reference_candidate_triage",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "raw_reference_candidate_triage_lists_paths_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "raw_reference_candidate_triage_never_starts_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def name_parts(path: Path) -> set[str]:
    text = re.sub(r"[^a-z0-9]+", " ", path.stem.lower())
    return {part for part in text.split() if part}


def normalized_name(path: Path) -> str:
    return " ".join(sorted(name_parts(path)))


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def audio_files(candidate_roots: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for root in candidate_roots:
        if is_audio_file(root):
            result.append(root)
        elif root.is_dir():
            result.extend(path for path in root.rglob("*") if is_audio_file(path))
    return sorted(set(result), key=lambda path: str(path).replace("\\", "/").lower())


def group_audio_by_directory(paths: Sequence[Path]) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.parent].append(path)
    return {directory: sorted(files) for directory, files in grouped.items()}


def mix_score(path: Path) -> int:
    parts = name_parts(path)
    if {"source", "mix"}.issubset(parts):
        return 100
    if {"full", "mix"}.issubset(parts):
        return 95
    if "input" in parts and ("mix" in parts or "full" in parts):
        return 90
    if parts in ({"mix"}, {"source"}, {"input"}):
        return 80
    return 0


def select_mix(files: Sequence[Path]) -> Path | None:
    scored = [(mix_score(path), path) for path in files]
    scored = [(score, path) for score, path in scored if score > 0]
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[0], str(item[1]).lower()))[0][1]


def path_key(path: Path) -> str:
    try:
        normalized = path.resolve(strict=False)
    except OSError:
        normalized = path
    return str(normalized).replace("\\", "/").lower()


def same_path(left: Path, right: Path) -> bool:
    return path_key(left) == path_key(right)


def is_under_or_same(path: Path, root: Path) -> bool:
    path_text = path_key(path).rstrip("/")
    root_text = path_key(root).rstrip("/")
    return path_text == root_text or path_text.startswith(root_text + "/")


def resolve_manifest_path(manifest_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def read_pack_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def listen_pack_manifest_is_usable(payload: dict[str, Any]) -> bool:
    if payload.get("schema") not in {AGGREGATE_PACK_SCHEMA, STEM_PACK_SCHEMA}:
        return False
    if payload.get("delivery_status") != "accepted":
        return False
    if payload.get("delivery_allowed") is not True:
        return False
    if payload.get("presentation_status") != "accepted_for_listen_pack":
        return False
    if payload.get("release_ready") is True or payload.get("can_publish_as_result") is True:
        return False
    return not payload.get("failed_checks")


def manifest_audio_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    key = "items" if payload.get("schema") == AGGREGATE_PACK_SCHEMA else "stems"
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def row_listen_audio_paths(manifest_path: Path, row: dict[str, Any]) -> list[Path]:
    result: list[Path] = []
    for key in ("audio_path", "artifact_relative_path", "listen_audio_path"):
        raw = row.get(key)
        if not raw:
            continue
        path = resolve_manifest_path(manifest_path, raw)
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            result.append(path)
    return result


def matching_manifest_rows(manifest_path: Path, payload: dict[str, Any], files: Sequence[Path]) -> list[dict[str, Any]]:
    file_keys = {path_key(path) for path in files}
    rows: list[dict[str, Any]] = []
    for row in manifest_audio_rows(payload):
        if any(path_key(path) in file_keys for path in row_listen_audio_paths(manifest_path, row)):
            rows.append(row)
    return rows


def valid_source_mix_path(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file()


def manifest_source_audio_path(manifest_path: Path, payload: dict[str, Any]) -> Path | None:
    raw_path = payload.get("source_audio_path")
    if not raw_path:
        return None
    path = resolve_manifest_path(manifest_path, raw_path)
    return path if valid_source_mix_path(path) else None


def referenced_source_manifest_paths(manifest_path: Path, payload: dict[str, Any], rows: Sequence[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for row in rows:
        raw_path = row.get("source_manifest")
        if raw_path:
            paths.append(resolve_manifest_path(manifest_path, raw_path))
    if not paths:
        source_manifests = payload.get("source_manifests")
        if isinstance(source_manifests, list) and len(source_manifests) == 1:
            raw_path = source_manifests[0].get("manifest_path") if isinstance(source_manifests[0], dict) else ""
            if raw_path:
                paths.append(resolve_manifest_path(manifest_path, raw_path))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def manifest_source_mix(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
    matching_rows: Sequence[dict[str, Any]],
) -> Path | None:
    if payload.get("schema") == STEM_PACK_SCHEMA:
        return manifest_source_audio_path(manifest_path, payload)
    candidates: list[Path] = []
    direct = manifest_source_audio_path(manifest_path, payload)
    if direct is not None:
        candidates.append(direct)
    for source_manifest_path in referenced_source_manifest_paths(manifest_path, payload, matching_rows):
        source_payload = read_pack_manifest(source_manifest_path)
        if not source_payload or not listen_pack_manifest_is_usable(source_payload):
            continue
        source_mix = manifest_source_audio_path(source_manifest_path, source_payload)
        if source_mix is not None:
            candidates.append(source_mix)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique[0] if len(unique) == 1 else None


def ancestor_pack_manifests(directory: Path, candidate_roots: Sequence[Path]) -> list[Path]:
    roots = [root if root.is_dir() else root.parent for root in candidate_roots]
    manifests: list[Path] = []
    current = directory
    while True:
        if roots and not any(is_under_or_same(current, root) for root in roots):
            break
        manifest_path = current / PACK_MANIFEST_NAME
        if manifest_path.is_file():
            manifests.append(manifest_path)
        if current.parent == current or any(same_path(current, root) for root in roots):
            break
        current = current.parent
    return manifests


def select_manifest_source_mix(
    directory: Path,
    files: Sequence[Path],
    *,
    candidate_roots: Sequence[Path],
) -> tuple[Path | None, str]:
    for manifest_path in ancestor_pack_manifests(directory, candidate_roots):
        payload = read_pack_manifest(manifest_path)
        if not payload or not listen_pack_manifest_is_usable(payload):
            continue
        rows = matching_manifest_rows(manifest_path, payload, files)
        if not rows:
            continue
        source_mix = manifest_source_mix(manifest_path=manifest_path, payload=payload, matching_rows=rows)
        if source_mix is not None:
            return source_mix, str(manifest_path)
    return None, ""


def matches_stem(path: Path, stem: str) -> bool:
    parts = name_parts(path)
    text = normalized_name(path)
    for token in STEM_TOKENS.get(stem, (stem,)):
        token_text = token.lower()
        if " " in token_text:
            if token_text in text:
                return True
        elif token_text in parts:
            return True
    return False


def stem_candidates(files: Sequence[Path], *, mix_path: Path | None, required_stems: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for stem in required_stems:
        normalized = normalized_stem(stem)
        candidates = [path for path in files if path != mix_path and matches_stem(path, normalized)]
        if candidates:
            result[normalized] = sorted(candidates, key=lambda path: str(path).lower())[0]
    return result


def rejected_directory(directory: Path, files: Sequence[Path], *, reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "full_pack_raw_reference_candidate_rejection.v1",
        "directory": str(directory),
        "audio_file_count": len(files),
        "audio_files": [str(path) for path in files],
        "rejection_reasons": list(reasons),
        "delivery_allowed": False,
        "release_ready": False,
        "training_started": False,
    }


def candidate_row(
    *,
    directory: Path,
    mix_path: Path,
    stem: str,
    stem_path: Path,
    source_mix_resolution: str = "same_directory_audio",
    source_mix_manifest_json: str = "",
) -> dict[str, Any]:
    return {
        "schema": "full_pack_raw_reference_candidate_row.v1",
        "id": stable_hash(["full_pack_raw_reference_candidate", str(directory), stem, str(stem_path)])[:16],
        "triage_status": "needs_verified_ground_truth_manifest",
        "stem_type": stem,
        "track_id_hint": directory.name,
        "candidate_directory": str(directory),
        "source_mix_candidate_path": str(mix_path),
        "source_mix_resolution": source_mix_resolution,
        "source_mix_manifest_json": source_mix_manifest_json,
        "target_stem_candidate_path": str(stem_path),
        "required_bundle_files": ["source_mix.flac", f"stems/{stem}.flac", "ground_truth_manifest.json"],
        "missing_acceptance_evidence": [
            "ground_truth_manifest.json",
            "ground_truth_contract.verified=true",
            "trainable_rights_or_project_owned_permission",
            "reference_waveform_audit_ready",
        ],
        "accepted_reference_row": False,
        "eligible_reference_material": False,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "training_started": False,
    }


def build_summary(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Pack Raw Reference Candidate Audit",
            "",
            f"- raw_candidate_status: `{manifest['raw_candidate_status']}`",
            f"- raw_candidate_rows: `{manifest['raw_candidate_rows']}`",
            f"- rejected_directory_count: `{manifest['rejected_directory_count']}`",
            f"- accepted_reference_rows: `{manifest['accepted_reference_rows']}`",
            f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
            "",
        ]
    )


def audit_full_pack_raw_reference_candidates(
    *,
    gap_plan_json: Path,
    candidate_roots: Sequence[Path],
    output_dir: Path,
    progress_every: int = 500,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", gap_plan_json=str(gap_plan_json), candidate_roots=[str(path) for path in candidate_roots])

    gap_rows, failed = load_gap_rows(gap_plan_json)
    required_stems = sorted(required_stems_from_gaps(gap_rows)) if not failed else []
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    grouped = group_audio_by_directory(audio_files(candidate_roots)) if not failed else {}
    sorted_groups = sorted(grouped.items(), key=lambda item: str(item[0]).lower())
    heartbeat_every = max(int(progress_every or 0), 1)
    for directory_index, (directory, files) in enumerate(sorted_groups, start=1):
        mix_path = select_mix(files)
        source_mix_resolution = "same_directory_audio"
        source_mix_manifest_json = ""
        if mix_path is None:
            mix_path, source_mix_manifest_json = select_manifest_source_mix(
                directory,
                files,
                candidate_roots=candidate_roots,
            )
            if mix_path is not None:
                source_mix_resolution = "pack_manifest_source_audio_path"
        if mix_path is None:
            rejected.append(rejected_directory(directory, files, reasons=["missing_source_mix_candidate"]))
        else:
            stems = stem_candidates(files, mix_path=mix_path, required_stems=required_stems)
            if not stems:
                rejected.append(rejected_directory(directory, files, reasons=["missing_required_stem_candidate"]))
            else:
                for stem, stem_path in sorted(stems.items()):
                    rows.append(
                        candidate_row(
                            directory=directory,
                            mix_path=mix_path,
                            stem=stem,
                            stem_path=stem_path,
                            source_mix_resolution=source_mix_resolution,
                            source_mix_manifest_json=source_mix_manifest_json,
                        )
                    )
        if directory_index % heartbeat_every == 0 or directory_index == len(sorted_groups):
            progress(
                progress_path,
                "scan_directory_progress",
                directories_seen=directory_index,
                audio_directory_count=len(sorted_groups),
                raw_candidate_rows=len(rows),
                rejected_directory_count=len(rejected),
                current_directory=str(directory),
            )

    rows.sort(key=lambda row: (row["stem_type"], row["track_id_hint"], row["id"]))
    by_stem = Counter(row["stem_type"] for row in rows)
    if failed:
        status = "blocked"
    elif rows:
        status = "raw_reference_candidates_need_verified_manifest"
    else:
        status = "blocked_no_raw_reference_candidates"

    manifest = {
        "schema": "full_pack_raw_reference_candidates_audit.v1",
        "manifest_type": "full_pack_raw_reference_candidates_audit",
        "raw_candidate_status": status,
        "failed_checks": failed,
        "gap_plan_json": str(gap_plan_json),
        "candidate_roots": [str(path) for path in candidate_roots],
        "audio_directory_count": len(grouped),
        "raw_candidate_rows": len(rows),
        "rejected_directory_count": len(rejected),
        "raw_candidates_by_stem": dict(sorted(by_stem.items())),
        "accepted_reference_rows": 0,
        "eligible_reference_material_rows": 0,
        "outputs": {
            "manifest_json": str(output_dir / MANIFEST_NAME),
            "rows_jsonl": str(output_dir / ROWS_NAME),
            "rejected_jsonl": str(output_dir / REJECTED_NAME),
            "progress_jsonl": str(progress_path),
            "summary_md": str(output_dir / SUMMARY_NAME),
        },
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "delivery_allowed": False,
        "presentation_status": "raw_reference_candidate_audit_not_for_listen_pack",
        "release_status": "diagnostic_artifact_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "safety": {
            "metadata_only": True,
            "path_listing_only": True,
            "audio_read": False,
            "audio_copied": False,
            "audio_decoded": False,
            "dataset_built": False,
            "training_started": False,
            "ground_truth_manifest_written": False,
            "audio_placeholder_written": False,
            "not_release_artifact": True,
            "accepted_rows_not_imported": True,
        },
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_jsonl(output_dir / REJECTED_NAME, rejected)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(build_summary(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        raw_candidate_status=status,
        raw_candidate_rows=len(rows),
        rejected_directory_count=len(rejected),
        failed_checks=failed,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage raw audio folders that might become full-pack reference bundles.")
    parser.add_argument("--gap-plan-json", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="audit_full_pack_raw_reference_candidates.py",
        entrypoint_path="/app/tools/audit_full_pack_raw_reference_candidates.py",
    )
    args = parse_args(argv)
    manifest = audit_full_pack_raw_reference_candidates(
        gap_plan_json=args.gap_plan_json,
        candidate_roots=args.candidate_root,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "raw_candidate_status": manifest["raw_candidate_status"],
                "raw_candidate_rows": manifest["raw_candidate_rows"],
                "accepted_reference_rows": manifest["accepted_reference_rows"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
