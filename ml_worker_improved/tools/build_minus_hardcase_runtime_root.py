#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "fine_tune_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "models_changed": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "audio_exported": False,
    "audio_copied": False,
    "release_ready": False,
}

FORBIDDEN_ACTIONS = (
    "present_as_listen_result",
    "mark_release_ready",
    "start_training",
    "relax_gates",
    "copy_audio_into_runtime_root",
)

OWNED_OUTPUT_FILES = (
    "run_manifest.json",
    "candidate_pool.jsonl",
    "selected_candidates.jsonl",
    "pair_compatibility.jsonl",
    "preview_render_contract.json",
    "summary.md",
    "skipped.jsonl",
    "progress.jsonl",
)

DEFAULT_MAC_PROJECT_ROOT = os.environ.get("AIMUSIC_MAC_PROJECT_ROOT", "/Users/andrejviluha/AIMusicProject_win")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(output_dir: Path, phase: str, **fields: Any) -> None:
    row = {"phase": phase, **fields, "release_ready": False}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_path_rewrite(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path rewrite must use FROM=TO")
    source, target = value.split("=", 1)
    if not source:
        raise argparse.ArgumentTypeError("path rewrite FROM is empty")
    return source.rstrip("/\\"), target.rstrip("/\\")


def default_path_rewrites(mac_project_root: str) -> list[tuple[str, str]]:
    root = mac_project_root.rstrip("/\\")
    return [
        ("/Users/andrejviluha/aimusic_local_artifacts", f"{root}/artifacts"),
        ("/host_artifacts", f"{root}/artifacts"),
        ("D:\\AIMusicProject", root),
        ("D:/AIMusicProject", root),
    ]


def normalize_path(value: Any, rewrites: Sequence[tuple[str, str]]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/") if text[:3].lower() in {"d:/", "d:\\"} else text
    for source, target in rewrites:
        source_norm = source.replace("\\", "/")
        normalized_norm = normalized.replace("\\", "/")
        if normalized_norm == source_norm or normalized_norm.startswith(source_norm + "/"):
            suffix = normalized_norm[len(source_norm) :].lstrip("/")
            return f"{target.rstrip('/')}/{suffix}" if suffix else target.rstrip("/")
    return normalized


def normalize_nested_paths(value: Any, rewrites: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_nested_paths(child, rewrites) for key, child in value.items()}
    if isinstance(value, list):
        return [normalize_nested_paths(child, rewrites) for child in value]
    if isinstance(value, str):
        return normalize_path(value, rewrites)
    return value


def path_exists(text: str) -> bool:
    return bool(text) and Path(text).exists()


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def stem_path(payload: dict[str, Any], *names: str) -> str:
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        return ""
    for name in names:
        value = stems.get(name)
        if isinstance(value, dict) and value.get("path"):
            return str(value["path"])
        if isinstance(value, str):
            return value
    return ""


def stem_entry(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    stems = payload.get("stems")
    if not isinstance(stems, dict):
        return {}
    for name in names:
        value = stems.get(name)
        if isinstance(value, dict) and value.get("path"):
            return dict(value)
        if isinstance(value, str):
            return {"path": value}
    return {}


def normalize_target_stem(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"instrumental", "instrumentals", "minus"}:
        return "instrumental"
    if text in {"vocal", "vocals", "voice"}:
        return "vocals"
    return text


def vocal_entry_is_companion(payload: dict[str, Any], entry: dict[str, Any]) -> bool:
    if entry.get("residual_companion") is True:
        return True
    return normalize_target_stem(payload.get("target_stem")) == "instrumental"


def first_present(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def status_candidate_id(manifest_path: Path) -> str:
    status_path = manifest_path.with_name("status.json")
    if not status_path.exists():
        return ""
    try:
        return str(read_json(status_path).get("candidate_id") or "")
    except Exception:
        return ""


def normalize_candidate_row(
    row: dict[str, Any],
    *,
    source_manifest: Path,
    rewrites: Sequence[tuple[str, str]],
    validate_audio: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mix_path = normalize_path(row.get("mix_path") or row.get("source_audio_path") or row.get("raw_audio_path"), rewrites)
    vocal_path = normalize_path(row.get("vocal_path") or row.get("selected_vocal_path"), rewrites)
    vocal_companion_path = normalize_path(row.get("vocal_companion_path") or row.get("vocal_analysis_path"), rewrites)
    vocal_analysis_path = vocal_path or vocal_companion_path
    minus_path = normalize_path(
        row.get("minus_path") or row.get("instrumental_path") or row.get("selected_minus_path") or row.get("rendered_minus_candidate_path"),
        rewrites,
    )
    candidate_id = str(row.get("candidate_id") or f"minus_hc_{stable_hash(str(source_manifest))[:12]}")
    track_id = str(row.get("track_id") or stable_hash(mix_path or source_manifest.parent.name)[:12])
    route_id = str(row.get("route_id") or row.get("minus_route_id") or row.get("model_id") or row.get("candidate_config_id") or candidate_id)
    reasons: list[str] = []
    for field, value in (("mix_path", mix_path), ("vocal_analysis_path", vocal_analysis_path), ("minus_path", minus_path)):
        if not value:
            reasons.append(f"missing_{field}")
        elif validate_audio and not path_exists(value):
            reasons.append(f"missing_audio_file:{field}")
    if reasons:
        return None, {
            "source_manifest": str(source_manifest),
            "track_id": track_id,
            "candidate_id": candidate_id,
            "skip_reason": ";".join(reasons),
            "release_ready": False,
        }
    normalized = {
        **row,
        "track_id": track_id,
        "run_id": str(row.get("run_id") or ""),
        "candidate_id": candidate_id,
        "route_id": route_id,
        "candidate_config_id": str(row.get("candidate_config_id") or route_id),
        "model_id": str(row.get("model_id") or row.get("minus_model_id") or route_id),
        "minus_model_id": str(row.get("minus_model_id") or row.get("model_id") or route_id),
        "candidate_family": str(row.get("candidate_family") or row.get("source_candidate_family") or "minus_hardcase_manifest"),
        "source_candidate_family": str(row.get("source_candidate_family") or row.get("candidate_family") or "minus_hardcase_manifest"),
        "mix_path": mix_path,
        "source_audio_path": mix_path,
        "vocal_path": vocal_path,
        "selected_vocal_path": vocal_path,
        "vocal_companion_path": vocal_companion_path,
        "vocal_companion_role": str(row.get("vocal_companion_role") or ""),
        "vocal_companion_source": str(row.get("vocal_companion_source") or ""),
        "vocal_companion_model_id": str(row.get("vocal_companion_model_id") or ""),
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "selected_minus_path": minus_path,
        "stem_paths": normalize_nested_paths(row.get("stem_paths") or {}, rewrites),
        "stem_type": "minus",
        "score_mode": "minus",
        "is_complete_pair": True,
        "can_use_for_minus": True,
        "source_manifest": str(source_manifest),
        "candidate_manifest": str(row.get("candidate_manifest") or row.get("source_manifest") or source_manifest),
        "delivery_allowed": False,
        "presentation_status": "diagnostic_runtime_root_candidate_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_result": False,
        "forbidden_actions": sorted(set(FORBIDDEN_ACTIONS) | {str(item) for item in row.get("forbidden_actions", []) or []}),
        **SAFETY_FLAGS,
    }
    return normalized, None


def row_from_manifest(
    manifest_path: Path,
    *,
    rewrites: Sequence[tuple[str, str]],
    validate_audio: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = read_json(manifest_path)
    if isinstance(payload.get("candidate_row"), dict):
        return normalize_candidate_row(
            dict(payload["candidate_row"]),
            source_manifest=manifest_path,
            rewrites=rewrites,
            validate_audio=validate_audio,
        )

    mix_path = first_present(
        nested_get(payload, "artifacts", "raw_ref_flac"),
        nested_get(payload, "artifacts", "raw_ref_audio"),
        nested_get(payload, "audio", "raw_ref"),
        payload.get("raw_audio_path"),
        payload.get("source_audio_path"),
    )
    vocal_entry = stem_entry(payload, "vocals", "vocal")
    vocal_path = first_present(
        vocal_entry.get("path"),
        nested_get(payload, "artifacts", "vocals_flac"),
        nested_get(payload, "audio", "vocals"),
        payload.get("vocal_path"),
    )
    minus_path = first_present(
        stem_path(payload, "instrumental", "minus"),
        nested_get(payload, "artifacts", "instrumental_flac"),
        nested_get(payload, "audio", "instrumental"),
        payload.get("minus_path"),
        payload.get("rendered_minus_candidate_path"),
    )
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    model_id = str(payload.get("model_id") or payload.get("model_name") or model.get("model_id") or manifest_path.parent.name)
    row = {
        "track_id": str(payload.get("track_id") or ""),
        "run_id": str(payload.get("run_id") or payload.get("artifact_id") or manifest_path.parent.name),
        "candidate_id": status_candidate_id(manifest_path) or str(payload.get("candidate_id") or f"{manifest_path.parent.name}__minus__{model_id}"),
        "route_id": str(payload.get("route_id") or payload.get("candidate_config_id") or model_id),
        "model_id": model_id,
        "minus_model_id": model_id,
        "candidate_family": str(model.get("family") or payload.get("candidate_family") or "minus_hardcase_manifest"),
        "source_candidate_family": str(model.get("family") or payload.get("candidate_family") or "minus_hardcase_manifest"),
        "mix_path": mix_path,
        "source_audio_path": mix_path,
        "vocal_path": "" if vocal_entry_is_companion(payload, vocal_entry) else vocal_path,
        "selected_vocal_path": "" if vocal_entry_is_companion(payload, vocal_entry) else vocal_path,
        "vocal_companion_path": vocal_path if vocal_entry_is_companion(payload, vocal_entry) else "",
        "vocal_companion_role": "residual_companion" if vocal_entry.get("residual_companion") is True else ("minus_route_vocal_companion" if vocal_entry_is_companion(payload, vocal_entry) else ""),
        "vocal_companion_source": str(vocal_entry.get("source") or ""),
        "vocal_companion_model_id": model_id if vocal_entry_is_companion(payload, vocal_entry) else "",
        "minus_path": minus_path,
        "instrumental_path": minus_path,
        "selected_minus_path": minus_path,
        "stem_paths": payload.get("stems") if isinstance(payload.get("stems"), dict) else {},
    }
    return normalize_candidate_row(row, source_manifest=manifest_path, rewrites=rewrites, validate_audio=validate_audio)


def rows_from_jsonl(
    path: Path,
    *,
    rewrites: Sequence[tuple[str, str]],
    validate_audio: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        normalized, skip = normalize_candidate_row(row, source_manifest=path, rewrites=rewrites, validate_audio=validate_audio)
        if normalized is not None:
            rows.append(normalized)
        if skip is not None:
            skipped.append(skip)
    return rows, skipped


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in sorted(rows, key=lambda item: (str(item["track_id"]), str(item["candidate_id"]), str(item["minus_path"]))):
        key = (str(row["track_id"]), str(row["candidate_id"]), str(row["minus_path"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Minus Hardcase Runtime Root",
        "",
        f"- run_id: {manifest['run_id']}",
        f"- candidate_count: {manifest['candidate_count']}",
        f"- skipped_count: {manifest['skipped_count']}",
        "- delivery_allowed: false",
        "- release_ready: false",
        f"- next_step: {manifest['next_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OWNED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def build_minus_hardcase_runtime_root(
    *,
    candidate_manifests: Sequence[Path],
    candidate_jsonls: Sequence[Path],
    output_dir: Path,
    path_rewrites: Sequence[tuple[str, str]],
    validate_audio: bool = True,
    command_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    progress(output_dir, "start", candidate_manifests=[str(path) for path in candidate_manifests], candidate_jsonls=[str(path) for path in candidate_jsonls])
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for manifest_path in candidate_manifests:
        progress(output_dir, "candidate_manifest_start", path=str(manifest_path))
        row, skip = row_from_manifest(manifest_path, rewrites=path_rewrites, validate_audio=validate_audio)
        if row is not None:
            rows.append(row)
        if skip is not None:
            skipped.append(skip)
        progress(output_dir, "candidate_manifest_done", path=str(manifest_path), emitted=row is not None, skipped=skip is not None)
    for jsonl_path in candidate_jsonls:
        progress(output_dir, "candidate_jsonl_start", path=str(jsonl_path))
        jsonl_rows, jsonl_skipped = rows_from_jsonl(jsonl_path, rewrites=path_rewrites, validate_audio=validate_audio)
        rows.extend(jsonl_rows)
        skipped.extend(jsonl_skipped)
        progress(output_dir, "candidate_jsonl_done", path=str(jsonl_path), emitted=len(jsonl_rows), skipped=len(jsonl_skipped))

    rows = dedupe_rows(rows)
    run_id = output_dir.name
    for row in rows:
        row["run_id"] = str(row.get("run_id") or run_id)
    track_ids = sorted({str(row.get("track_id") or "") for row in rows if row.get("track_id")})
    manifest = {
        "schema": "minus_hardcase_runtime_root.v1",
        "run_id": run_id,
        "track_ids": track_ids,
        "track_count": len(track_ids),
        "candidate_count": len(rows),
        "skipped_count": len(skipped),
        "output_status": "diagnostic_runtime_root_ready" if rows else "blocked_no_complete_candidates",
        "next_step": "run_no_reference_windowed_judge_minus_then_audio_error_grounding" if rows else "fix_candidate_inputs",
        "candidate_pool_jsonl": str(output_dir / "candidate_pool.jsonl"),
        "minus_windowed_judge_jsonl_expected": str(output_dir / "minus_windowed_judge.jsonl"),
        "grounding_builder_compatible": bool(rows),
        "delivery_allowed": False,
        "presentation_status": "diagnostic_runtime_root_not_for_listen_pack",
        "release_status": "not_release_ready",
        "release_ready": False,
        "command_args": list(command_args or []),
        "path_rewrites": [{"from": source, "to": target} for source, target in path_rewrites],
        **SAFETY_FLAGS,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_jsonl(output_dir / "candidate_pool.jsonl", rows)
    write_jsonl(output_dir / "selected_candidates.jsonl", [])
    write_jsonl(output_dir / "pair_compatibility.jsonl", [])
    write_jsonl(output_dir / "skipped.jsonl", skipped)
    write_json(output_dir / "preview_render_contract.json", {"release_ready": False, "audio_preview_ready": False, **SAFETY_FLAGS})
    write_summary(output_dir / "summary.md", manifest)
    progress(output_dir, "complete", candidate_count=len(rows), skipped_count=len(skipped), manifest_path=str(output_dir / "run_manifest.json"))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a diagnostic minus hardcase runtime root from existing candidate artifacts.")
    parser.add_argument("--candidate-manifest", type=Path, action="append", default=[])
    parser.add_argument("--candidate-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mac-project-root", default=DEFAULT_MAC_PROJECT_ROOT)
    parser.add_argument("--path-rewrite", type=parse_path_rewrite, action="append", default=[])
    parser.add_argument("--no-validate-audio", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.candidate_manifest and not args.candidate_jsonl:
        raise SystemExit("at least one --candidate-manifest or --candidate-jsonl is required")
    rewrites = [*default_path_rewrites(args.mac_project_root), *args.path_rewrite]
    manifest = build_minus_hardcase_runtime_root(
        candidate_manifests=args.candidate_manifest,
        candidate_jsonls=args.candidate_jsonl,
        output_dir=args.output_dir,
        path_rewrites=rewrites,
        validate_audio=not args.no_validate_audio,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if manifest["candidate_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
