#!/usr/bin/env python3
"""Filter an accepted listen source manifest into a short-name subset pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import ARTIFACT_AUDIO_DIR, indexed_audio_filename, validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
EXCLUDED_NAME = "excluded.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"

BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "can_publish_as_release",
    "can_mark_release_ready",
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
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
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
        "expected_acceleration_mode": "external_only_cpu_subset_packaging",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "subset_pack_copies_existing_audio_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "subset_pack_no_training",
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


def source_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("listen_audio_files", "stems", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def row_route_intent(row: dict[str, Any]) -> str:
    if row.get("route_intent"):
        return str(row.get("route_intent"))
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    return str(source_row.get("route_intent") or "")


def row_audio_path(row: dict[str, Any], source_manifest_json: Path) -> Path:
    raw = str(row.get("listen_audio_path") or row.get("audio_path") or row.get("source_audio_path") or "")
    if not raw:
        raise FileNotFoundError("row_audio_path_missing")
    path = Path(raw)
    if not path.is_absolute():
        path = source_manifest_json.parent / path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def row_is_accepted(row: dict[str, Any]) -> bool:
    return (
        row.get("delivery_status", "accepted") == "accepted"
        and row.get("delivery_allowed", True) is True
        and row.get("release_ready") is not True
        and row.get("can_publish_as_result") is not True
        and row.get("can_publish_as_release") is not True
        and row.get("can_mark_release_ready") is not True
    )


def row_matches(row: dict[str, Any], include_stems: set[str], include_route_intents: set[str]) -> bool:
    stem_ok = not include_stems or str(row.get("stem_type") or "") in include_stems
    route_ok = not include_route_intents or row_route_intent(row) in include_route_intents
    return stem_ok and route_ok


def row_short_audio_path(row: dict[str, Any]) -> str:
    for key in ("artifact_relative_path", "audio_path", "source_artifact_relative_path"):
        raw = str(row.get(key) or "")
        if raw and not Path(raw).is_absolute():
            return Path(raw).as_posix()
    return ""


def minus_audit_rows(minus_audit_json: Path | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    if minus_audit_json is None:
        return {}, {}, []
    manifest = read_json(minus_audit_json)
    failed = collect_true_flag_failures("minus_audit", manifest)
    if manifest.get("schema") != "short_minus_quality_audit.v1":
        failed.append("minus_audit_schema_unexpected")
    rows_jsonl = Path(str(manifest.get("rows_jsonl") or ""))
    if not str(rows_jsonl):
        failed.append("minus_audit_rows_jsonl_missing")
        return manifest, {}, failed
    if not rows_jsonl.is_absolute():
        rows_jsonl = minus_audit_json.parent / rows_jsonl
    if not rows_jsonl.exists():
        failed.append("minus_audit_rows_jsonl_missing")
        return manifest, {}, failed
    by_path: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(rows_jsonl):
        failed.extend(collect_true_flag_failures("minus_audit_row", row))
        short_path = str(row.get("short_audio_path") or "")
        if short_path:
            by_path[Path(short_path).as_posix()] = row
    if not by_path:
        failed.append("minus_audit_rows_empty")
    return manifest, by_path, stable_unique(failed)


def filter_minus_audit_candidates(
    candidates: Sequence[dict[str, Any]],
    audit_by_path: dict[str, dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], list[dict[str, Any]], list[str]]:
    selected: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    excluded: list[dict[str, Any]] = []
    failed: list[str] = []
    for row in candidates:
        if str(row.get("stem_type") or "").lower() != "minus":
            selected.append((row, None))
            continue
        short_path = row_short_audio_path(row)
        audit_row = audit_by_path.get(short_path)
        if audit_row is None:
            failed.append(f"minus_audit_row_missing:{short_path or 'unknown'}")
            continue
        if audit_row.get("audit_decision") == "passed":
            selected.append((row, audit_row))
            continue
        blockers = list(audit_row.get("audit_blockers") or [])
        if not blockers:
            blockers = ["minus_audit_not_passed"]
        excluded.append(
            {
                "schema": "short_listen_subset_excluded_audio.v1",
                "source_artifact_relative_path": short_path,
                "stem_type": str(row.get("stem_type") or "minus"),
                "route_intent": row_route_intent(row),
                "audit_decision": str(audit_row.get("audit_decision") or "minus_audit_not_passed"),
                "audit_blockers": stable_unique(blockers),
                "quality_warning_reasons": list(audit_row.get("quality_warning_reasons") or []),
                "source_row": row,
                "release_ready": False,
                "can_publish_as_result": False,
            }
        )
    return selected, excluded, stable_unique(failed)


def stem_type_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stem = str(row.get("stem_type") or "unknown")
        counts[stem] = counts.get(stem, 0) + 1
    return dict(sorted(counts.items()))


def source_stem_type_counts(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        source_alias = str(row.get("source_alias") or "")
        stem = str(row.get("stem_type") or "unknown")
        source_counts = counts.setdefault(source_alias, {})
        source_counts[stem] = source_counts.get(stem, 0) + 1
    return {source: dict(sorted(source_counts.items())) for source, source_counts in sorted(counts.items())}


def copy_row(
    *,
    row: dict[str, Any],
    source_manifest_json: Path,
    output_dir: Path,
    alias_prefix: str,
    index: int,
    minus_audit_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_artifact_segment("sub")
    source_path = row_audio_path(row, source_manifest_json)
    filename = indexed_audio_filename(alias_prefix, index)
    target_path = output_dir / ARTIFACT_AUDIO_DIR / "sub" / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    metadata = audio_metadata(target_path)
    payload = {
        "schema": "short_listen_subset_audio.v1",
        "selection_rank": index,
        "route_intent": row_route_intent(row),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "source_alias": str(row.get("source_alias") or ""),
        "track_alias": str(row.get("track_alias") or ""),
        "artifact_alias": filename.removesuffix(".flac"),
        "artifact_filename": filename,
        "artifact_relative_path": str(target_path.relative_to(output_dir).as_posix()),
        "listen_audio_path": str(target_path),
        "listen_audio_sha256": sha256_file(target_path),
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "duration_sec": metadata["duration_sec"],
        "source_audio_path": str(source_path),
        "source_artifact_relative_path": str(row.get("artifact_relative_path") or row.get("audio_path") or ""),
        "source_candidate_id": str(row.get("source_candidate_id") or row.get("candidate_id") or ""),
        "source_row": row,
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "semantic_quality_verified": False,
    }
    if minus_audit_row is not None:
        warnings = list(minus_audit_row.get("quality_warning_reasons") or [])
        payload["minus_audit_gate"] = {
            "audit_decision": str(minus_audit_row.get("audit_decision") or ""),
            "audit_blockers": list(minus_audit_row.get("audit_blockers") or []),
            "quality_warning_reasons": warnings,
        }
        if warnings:
            payload["quality_warning_reasons"] = warnings
    return payload


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]], excluded_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short listen subset pack",
        "",
        f"- delivery_status: `{manifest['delivery_status']}`",
        f"- selected_count: `{manifest['selected_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Selected Audio",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['artifact_relative_path']}` `{row['stem_type']}` `{row['route_intent']}`")
    if excluded_rows:
        lines.extend(["", "## Excluded Audio", ""])
        for row in excluded_rows:
            lines.append(
                f"- `{row['source_artifact_relative_path']}` "
                f"`{row['stem_type']}` `{row['audit_decision']}` "
                f"`{','.join(row.get('audit_blockers') or [])}`"
            )
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_short_listen_subset_pack(
    *,
    source_manifest_json: Path,
    output_dir: Path,
    include_stems: Sequence[str] = (),
    include_route_intents: Sequence[str] = (),
    alias_prefix: str = "s",
    max_items: int = 0,
    minus_audit_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", source_manifest_json=str(source_manifest_json), minus_audit_json=str(minus_audit_json or ""))

    validate_artifact_segment("sub")
    validate_artifact_segment(indexed_audio_filename(alias_prefix, 1))
    source_manifest = read_json(source_manifest_json)
    audit_manifest, audit_by_path, audit_failed = minus_audit_rows(minus_audit_json)
    failed = collect_true_flag_failures("source", source_manifest)
    failed.extend(audit_failed)
    if source_manifest.get("delivery_status") != "accepted":
        failed.append("source_delivery_status_not_accepted")
    if source_manifest.get("delivery_allowed") is not True:
        failed.append("source_delivery_allowed_not_true")
    if source_manifest.get("presentation_status") != "accepted_for_listen_pack":
        failed.append("source_presentation_status_not_accepted")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    if not failed:
        stems = {str(value) for value in include_stems if str(value)}
        routes = {str(value) for value in include_route_intents if str(value)}
        candidates = [row for row in source_rows(source_manifest) if row_is_accepted(row) and row_matches(row, stems, routes)]
        candidate_pairs: list[tuple[dict[str, Any], dict[str, Any] | None]]
        if minus_audit_json is not None:
            candidate_pairs, excluded, gate_failed = filter_minus_audit_candidates(candidates, audit_by_path)
            failed.extend(gate_failed)
        else:
            candidate_pairs = [(row, None) for row in candidates]
        if max_items > 0:
            candidate_pairs = candidate_pairs[:max_items]
        if not candidate_pairs and not failed:
            failed.append("no_matching_source_rows")
        for index, (row, audit_row) in enumerate(candidate_pairs, start=1):
            selected.append(
                copy_row(
                    row=row,
                    source_manifest_json=source_manifest_json,
                    output_dir=output_dir,
                    alias_prefix=alias_prefix,
                    index=index,
                    minus_audit_row=audit_row,
                )
            )

    failed = stable_unique(failed)
    accepted = bool(selected) and not failed
    if not accepted:
        selected = []
        excluded = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    accepted_minus_count = sum(1 for row in selected if str(row.get("stem_type") or "").lower() == "minus")
    excluded_minus_count = sum(1 for row in excluded if str(row.get("stem_type") or "").lower() == "minus")
    manifest = {
        "schema": "short_listen_subset_pack_manifest.v1",
        "delivery_status": "accepted" if accepted else "blocked",
        "delivery_allowed": accepted,
        "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "deliverable": False,
        "listenable": accepted,
        "selected_count": len(selected),
        "listen_audio_files": selected,
        "stem_type_counts": stem_type_counts(selected),
        "source_stem_type_counts": source_stem_type_counts(selected),
        "source_manifest_json": str(source_manifest_json),
        "include_stems": list(include_stems),
        "include_route_intents": list(include_route_intents),
        "minus_audit_gate": {
            "schema": "short_subset_minus_audit_gate.v1",
            "enabled": minus_audit_json is not None,
            "minus_audit_json": str(minus_audit_json or ""),
            "audit_status": str(audit_manifest.get("audit_status") or ""),
            "accepted_minus_count": accepted_minus_count,
            "excluded_minus_count": excluded_minus_count,
            "excluded_jsonl": str(output_dir / EXCLUDED_NAME),
        },
        "failed_checks": failed,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "excluded_jsonl": str(output_dir / EXCLUDED_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "group_dir_strategy": "sub",
            "filename_strategy": "indexed_candidate_alias",
            "raw_track_or_candidate_ids_in_path": False,
            "provenance_in_manifest": True,
        },
        "contract": {
            "audio_copied_without_modification": True,
            "selection_only": True,
            "no_inference_run": True,
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "release_ready_forced_false": True,
            "short_windows_safe_artifact_names": True,
            "dirty_stems_never_ground_truth": True,
            "blocked_minus_rows_excluded_from_listen_items": minus_audit_json is not None,
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "not_claimed": ["release_ready", "publishable_result", "audio_repaired", "semantic_ground_truth", "clean_minus_for_excluded_rows"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected)
    write_jsonl(output_dir / EXCLUDED_NAME, excluded)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, selected, excluded), encoding="utf-8")
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], selected_count=len(selected), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an accepted listen source manifest into a short-name subset pack.")
    parser.add_argument("--source-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-stem", action="append", default=[])
    parser.add_argument("--include-route-intent", action="append", default=[])
    parser.add_argument("--alias-prefix", default="s")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--minus-audit-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_subset_pack.py",
        entrypoint_path="/app/tools/build_short_listen_subset_pack.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_subset_pack(
        source_manifest_json=args.source_manifest_json,
        output_dir=args.output_dir,
        include_stems=args.include_stem,
        include_route_intents=args.include_route_intent,
        alias_prefix=str(args.alias_prefix),
        max_items=int(args.max_items),
        minus_audit_json=args.minus_audit_json,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "selected_count": manifest["selected_count"],
                "accepted_minus_count": manifest["minus_audit_gate"]["accepted_minus_count"],
                "excluded_minus_count": manifest["minus_audit_gate"]["excluded_minus_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
