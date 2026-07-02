from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.no_reference_candidate_job_registry import (  # noqa: E402
    CandidateJobConfig,
    REGISTERED_MODEL_ENTRYPOINT,
    build_candidate_job_registry,
    local_registered_entrypoint_exists,
    route_family_for_model,
)
from worker.model_registry import ModelRegistry, ModelRegistryEntry, default_model_registry  # noqa: E402


DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"
HTDEMUCS_ROUTE_FAMILIES = {"htdemucs_ft", "htdemucs_6s"}
FORBIDDEN_ROUTE_TOKENS = ("cleanup", "debleed", "denoise", "eq", "threshold", "cloud")
SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "separator_training_started": False,
    "model_downloaded": False,
    "model_registry_edited": False,
    "thresholds_relaxed": False,
    "cleanup_routes_added": False,
    "melody_transcription_ran": False,
    "recommendation_index_updated": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def unavailable_reason(parts: Iterable[str]) -> str:
    return ";".join(str(part) for part in parts if str(part))


def unsupported_cli_forbidden_reason(config: CandidateJobConfig) -> str:
    if config.forbidden_reason:
        return config.forbidden_reason
    reasons = []
    for part in str(config.unavailable_reason or "").split(";"):
        if part.startswith("unsupported_parameter:"):
            reasons.append(part.replace("unsupported_parameter:", ""))
    if reasons:
        return "unsupported_cli_parameter:" + ",".join(sorted(reasons))
    return ""


def forbidden_reason_for(config: CandidateJobConfig) -> str:
    unsupported = unsupported_cli_forbidden_reason(config)
    if unsupported:
        return unsupported
    haystack = " ".join(
        [
            config.candidate_config_id,
            config.candidate_family,
            config.source_model_id,
            config.source_tool,
        ]
    ).lower()
    for token in FORBIDDEN_ROUTE_TOKENS:
        if token in haystack:
            return token
    return ""


def normalize_registry_stem(stem: str) -> str:
    value = str(stem or "").strip().lower()
    if value in {"instrumental", "instrumentals"}:
        return "instrumental"
    if value in {"minus", "clean_minus"}:
        return "minus"
    return value


def entry_supports_minus(entry: ModelRegistryEntry) -> bool:
    stems = {normalize_registry_stem(stem) for stem in (*entry.target_stems, *entry.output_stems)}
    return bool(stems & {"minus", "instrumental"})


def route_type_for(config: CandidateJobConfig) -> str:
    if config.target_mode == "both":
        return "existing_model_both"
    if config.target_mode == "minus":
        return "existing_model_minus"
    return "existing_model_variant"


def entry_by_model_id(registry: ModelRegistry) -> dict[str, ModelRegistryEntry]:
    return {entry.model_id: entry for entry in registry.entries}


def route_row(
    *,
    config: CandidateJobConfig,
    registry: ModelRegistry,
    entrypoint_path: Path,
) -> dict[str, Any] | None:
    if not (config.produces_minus or config.produces_both or config.target_mode in {"minus", "both"}):
        return None
    entries = entry_by_model_id(registry)
    entry = entries.get(config.source_model_id)
    can_run_without_registry_edit = entry is not None and bool(config.can_run_without_registry_edit)
    missing_notes: list[str] = []
    model_files_available = False
    supports_minus = bool(config.produces_minus or config.produces_both or config.target_mode in {"minus", "both"})
    supports_vocal = bool(config.produces_vocal or config.produces_both or config.target_mode in {"vocal", "both"})
    supports_both = bool(supports_minus and supports_vocal)
    route_family = config.route_family or route_family_for_model(config.source_model_id)
    if entry is None:
        missing_notes.append("model_id_missing_from_registry")
    else:
        availability = registry.availability(entry)
        model_files_available = availability.status == "available"
        missing_notes.extend(f"missing_path:{path}" for path in availability.missing_paths)
        missing_notes.extend(f"missing_python_module:{module}" for module in availability.missing_python_modules)
        missing_notes.extend(f"runtime_blocker:{blocker}" for blocker in availability.runtime_blockers)
        supports_minus = supports_minus or entry_supports_minus(entry)
        route_family = route_family_for_model(entry.model_id)

    entrypoint_available = entrypoint_path.exists()
    forbidden_reason = forbidden_reason_for(config)
    can_run_without_download = bool(
        config.can_run_without_download
        and model_files_available
        and not any(note.startswith("missing_path:") for note in missing_notes)
    )
    notes = list(missing_notes)
    if not entrypoint_available:
        notes.append(f"missing_entrypoint:{REGISTERED_MODEL_ENTRYPOINT}")
    if config.unavailable_reason:
        notes.append(str(config.unavailable_reason))
    if forbidden_reason:
        notes.append(f"forbidden_route:{forbidden_reason}")
    available = bool(
        config.available
        and config.entrypoint_type != "unavailable"
        and entrypoint_available
        and model_files_available
        and can_run_without_registry_edit
        and can_run_without_download
        and supports_minus
        and not forbidden_reason
    )
    return {
        "route_id": config.route_id,
        "candidate_config_id": config.candidate_config_id,
        "route_type": route_type_for(config),
        "source_model_id": config.source_model_id,
        "source_tool": config.source_tool,
        "entrypoint": str(entrypoint_path),
        "entrypoint_available": bool(entrypoint_available),
        "model_files_available": bool(model_files_available),
        "supports_minus": bool(supports_minus),
        "supports_vocal": bool(supports_vocal),
        "supports_both": bool(supports_both),
        "supports_parameter_variants": bool(config.supported_parameters and not forbidden_reason),
        "supported_parameters": list(config.supported_parameters),
        "expected_outputs": dict(config.expected_outputs),
        "gpu_required": bool(config.gpu_required),
        "available": bool(available),
        "unavailable_reason": unavailable_reason(notes),
        "route_family": route_family,
        "can_run_without_download": bool(can_run_without_download),
        "can_run_without_registry_edit": bool(can_run_without_registry_edit),
        "forbidden_reason": forbidden_reason,
        "candidate_family": config.candidate_family,
        "target_mode": config.target_mode,
        "shared_execution_group_id": config.shared_execution_group_id,
        "produces_vocal": bool(config.produces_vocal),
        "produces_minus": bool(config.produces_minus),
        "produces_both": bool(config.produces_both),
        "materialized_candidate_modes": list(config.materialized_candidate_modes),
        "source_config_hash": config.source_config_hash,
        "created_at": DETERMINISTIC_CREATED_AT,
        **SAFETY_FLAGS,
    }


def build_summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    available_families = {str(row.get("route_family") or "") for row in routes if row.get("available") is True}
    quality_route_families = {family for family in available_families if family}
    route_space_limited = not quality_route_families or quality_route_families <= HTDEMUCS_ROUTE_FAMILIES
    return {
        "discovered_routes_total": len(routes),
        "available_routes_total": sum(1 for row in routes if row.get("available") is True),
        "unavailable_routes_total": sum(1 for row in routes if row.get("available") is not True),
        "route_space_limited": bool(route_space_limited),
        "route_families": sorted({str(row.get("route_family") or "") for row in routes if row.get("route_family")}),
        "available_route_families": sorted(quality_route_families),
        "forbidden_routes_total": sum(1 for row in routes if row.get("forbidden_reason")),
        "minus_routes_total": sum(1 for row in routes if row.get("supports_minus") is True),
        "both_routes_total": sum(1 for row in routes if row.get("supports_both") is True),
        **SAFETY_FLAGS,
    }


def discover_minus_routes(
    *,
    registry: ModelRegistry | None = None,
    registry_configs: Iterable[CandidateJobConfig] | None = None,
    entrypoint_path: Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    registry = registry or default_model_registry()
    configs = list(registry_configs) if registry_configs is not None else build_candidate_job_registry(profile=profile, registry=registry)
    if entrypoint_path is None:
        entrypoint_path = APP_ROOT / REGISTERED_MODEL_ENTRYPOINT
    rows = [
        row
        for config in configs
        for row in [route_row(config=config, registry=registry, entrypoint_path=entrypoint_path)]
        if row is not None
    ]
    rows.sort(key=lambda row: (str(row["route_id"]), str(row["candidate_config_id"])))
    return {
        "schema_version": 1,
        "created_at": DETERMINISTIC_CREATED_AT,
        "profile": profile or "",
        "source_files": [
            "worker/model_registry.py",
            "tools/no_reference_candidate_job_registry.py",
            REGISTERED_MODEL_ENTRYPOINT,
        ],
        "local_registered_entrypoint_exists": bool(local_registered_entrypoint_exists()),
        "routes": rows,
        "summary": build_summary(rows),
    }


def write_discovery_outputs(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = list(manifest.get("routes") or [])
    paths = {
        "json": output_dir / "minus_route_discovery.json",
        "jsonl": output_dir / "minus_route_discovery.jsonl",
        "summary_md": output_dir / "minus_route_discovery_summary.md",
    }
    write_json(paths["json"], manifest)
    write_jsonl(paths["jsonl"], routes)
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    lines = [
        "# No-Reference Minus Route Discovery",
        "",
        f"- discovered_routes_total: {summary.get('discovered_routes_total', 0)}",
        f"- available_routes_total: {summary.get('available_routes_total', 0)}",
        f"- unavailable_routes_total: {summary.get('unavailable_routes_total', 0)}",
        f"- route_space_limited: {str(summary.get('route_space_limited') is True).lower()}",
        f"- model_downloaded: {str(summary.get('model_downloaded') is True).lower()}",
        f"- model_registry_edited: {str(summary.get('model_registry_edited') is True).lower()}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        "",
        "## Available Routes",
    ]
    available = [row for row in routes if row.get("available") is True]
    if available:
        lines.extend(f"- {row.get('route_id')} ({row.get('route_family')})" for row in available)
    else:
        lines.append("- none")
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover existing no-reference minus routes without mutating registries.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("safe", "balanced", "exploration"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = discover_minus_routes(profile=args.profile)
    paths = write_discovery_outputs(args.output_dir, manifest)
    print(json.dumps({"minus_route_discovery_json": str(paths["json"]), **manifest["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
