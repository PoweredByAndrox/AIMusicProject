#!/usr/bin/env python3
"""Materialize router-selected registered minus model candidate commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from docker_runtime import cli_runtime_contract, require_docker_runtime  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


SCHEMA = "minus_delta_model_candidate_command_plan.v1"
ROUTE = "no_reference_minus_candidate_model_router"
TOOL = "/app/tools/run_registered_model_candidate.py"
MANIFEST_NAME = "plan.json"
ROWS_NAME = "rows.jsonl"
COMMAND_ROWS_NAME = "cmds.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
MODEL_OUTPUT_DIR_NAME = "mdl"
DEFAULT_MAX_CANDIDATES = 3
DOCKER_VISIBLE_PATH_PREFIXES = (
    "/app",
    "/host_artifacts",
    "/host_testing_stuff_music",
    "/host_testing_flstudio_music",
    "/host_downloads",
    "/main_lab_artifacts",
)

FAILURE_LABEL_EXPANSIONS = {
    "persistent_unsafe_vocal_or_full_mix_blockers": (
        "vocal_residue_in_minus",
        "reference_likeness_below_threshold",
    ),
    "failed_leakage_repair_no_blocker_reduction": ("vocal_residue_in_minus",),
    "vocal_projection_proxy": ("vocal_residue_in_minus",),
    "vocal_stem_corr_proxy": ("vocal_residue_in_minus",),
    "vocal_leakage_in_minus": ("vocal_residue_in_minus",),
    "full_mix_like_risk_proxy": ("reference_likeness_below_threshold",),
    "presence_loss_proxy": ("presence_loss",),
    "air_loss_proxy": ("presence_loss",),
    "low_rms_muffling_proxy": ("presence_loss",),
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def progress(path: Path, phase: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "phase": phase,
        "time_unix": round(time.time(), 6),
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "materializer writes registered model command plans only",
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def canonical_path(value: Any, base_dir: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def normalize_path_text(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def is_prefixed_path(path_text: str, prefix: str) -> bool:
    normalized_prefix = prefix.rstrip("/")
    return path_text == normalized_prefix or path_text.startswith(f"{normalized_prefix}/")


def docker_visible_path(value: Any) -> str | None:
    normalized = normalize_path_text(value)
    if not normalized:
        return None
    for prefix in DOCKER_VISIBLE_PATH_PREFIXES:
        if is_prefixed_path(normalized, prefix):
            return normalized
    host_artifacts_marker = "/host_artifacts/"
    if host_artifacts_marker in normalized:
        suffix = normalized.split(host_artifacts_marker, 1)[1].strip("/")
        return f"/host_artifacts/{suffix}" if suffix else "/host_artifacts"
    if normalized == "host_artifacts" or normalized.endswith("/host_artifacts"):
        return "/host_artifacts"
    return None


def docker_path_mapping_payload(
    *,
    source_audio_path: Path,
    output_dir: Path,
    docker_source_audio_path: str | None,
    docker_output_dir: str | None,
) -> dict[str, Any]:
    failed_path_fields: list[str] = []
    if not docker_source_audio_path:
        failed_path_fields.append("source_audio_path")
    if not docker_output_dir:
        failed_path_fields.append("output_dir")
    return {
        "source_audio_path": str(source_audio_path),
        "output_dir": str(output_dir),
        "docker_source_audio_path": docker_source_audio_path or "",
        "docker_output_dir": docker_output_dir or "",
        "failed_path_fields": failed_path_fields,
    }


def nested_get(payload: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_path(payload: dict[str, Any], base_dir: Path, candidates: Sequence[Sequence[str]]) -> Path | None:
    for keys in candidates:
        path = canonical_path(nested_get(payload, keys), base_dir)
        if path is not None:
            return path
    return None


def load_router_row(router_manifest: dict[str, Any], router_path: Path) -> dict[str, Any]:
    rows = router_manifest.get("rows")
    if isinstance(rows, list) and rows:
        return dict(rows[0])
    rows_path = canonical_path(router_manifest.get("rows_jsonl"), router_path.parent)
    if rows_path is not None and rows_path.exists():
        rows_from_file = list(iter_jsonl(rows_path))
        if rows_from_file:
            return dict(rows_from_file[0])
    return {}


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    text = str(value)
    return [text] if text else []


def issue_count_labels(*payloads: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for payload in payloads:
        summary = payload.get("issue_summary")
        if not isinstance(summary, dict):
            continue
        issue_counts = summary.get("issue_counts")
        if isinstance(issue_counts, dict):
            labels.extend(str(key) for key in issue_counts if str(key))
    return stable_unique(labels)


def expanded_failure_labels(router_manifest: dict[str, Any], router_row: dict[str, Any]) -> list[str]:
    base = stable_unique(
        as_string_list(router_manifest.get("failure_taxonomy"))
        + as_string_list(router_row.get("failure_taxonomy"))
        + issue_count_labels(router_manifest, router_row)
    )
    expanded: list[str] = []
    for label in base:
        expanded.append(label)
        expanded.extend(FAILURE_LABEL_EXPANSIONS.get(label, ()))
    return stable_unique(expanded)


def evaluated_labels(router_manifest: dict[str, Any], router_row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ("evaluated_model_ids", "evaluated_labels", "failed_model_ids"):
        labels.extend(as_string_list(router_manifest.get(key)))
        labels.extend(as_string_list(router_row.get(key)))
    return stable_unique(labels)


def suggestion_availability_dict(suggestion: Any) -> dict[str, Any]:
    availability = suggestion.availability
    return {
        "status": availability.status,
        "missing_paths": list(availability.missing_paths),
        "missing_python_modules": list(availability.missing_python_modules),
        "runtime_blockers": list(getattr(availability, "runtime_blockers", ())),
    }


def source_audio_path_from_evidence(
    *,
    router_plan_json: Path,
    source_repair_manifest: Path,
    router_manifest: dict[str, Any],
    router_row: dict[str, Any],
) -> Path | None:
    source_repair = read_json(source_repair_manifest)
    source_path = first_path(
        source_repair,
        source_repair_manifest.parent,
        (
            ("raw_audio_path",),
            ("raw_ref_path",),
            ("source_audio_path",),
            ("candidate", "raw_ref_path"),
            ("candidate", "source_audio_path"),
            ("inputs", "raw_ref_path"),
            ("inputs", "source_audio_path"),
        ),
    )
    if source_path is not None:
        return source_path

    delta_gate_path = canonical_path(
        router_row.get("delta_gate_manifest") or router_manifest.get("delta_gate_manifest"),
        router_plan_json.parent,
    )
    if delta_gate_path is None or not delta_gate_path.exists():
        return None
    delta_gate = read_json(delta_gate_path)
    after_manifest_path = canonical_path(delta_gate.get("after_self_judge_manifest"), delta_gate_path.parent)
    if after_manifest_path is None or not after_manifest_path.exists():
        return None
    after_manifest = read_json(after_manifest_path)
    return first_path(after_manifest, after_manifest_path.parent, (("source_audio_path",), ("raw_audio_path",)))


def summary_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Minus Delta Model Candidate Materializer",
        "",
        f"- command_plan_status: `{manifest['command_plan_status']}`",
        f"- selected_route: `{manifest.get('selected_route') or ''}`",
        f"- ready_command_count: `{manifest['ready_command_count']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        "",
        "## Failed Checks",
        "",
    ]
    failed_checks = manifest.get("failed_checks") or []
    lines.extend(f"- `{item}`" for item in failed_checks) if failed_checks else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def blocked_manifest(
    *,
    output_dir: Path,
    progress_path: Path,
    router_plan_json: Path,
    source_repair_manifest: Path,
    candidate_id: str,
    selected_route: str,
    status: str,
    failed_checks: Sequence[str],
    command_args: Sequence[str],
    runtime_environment: str = "docker",
    native_runtime_allowed: bool = False,
    source_failure_taxonomy: Sequence[str] = (),
    expanded_labels: Sequence[str] = (),
    docker_path_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA,
        "mode": "minus_delta_model_candidate_materializer",
        "command_plan_status": status,
        "candidate_id": candidate_id,
        "stem_type": "minus",
        "selected_route": selected_route,
        "ready_command_count": 0,
        "blocked_command_count": 1,
        "failed_checks": list(failed_checks),
        "router_plan_json": str(router_plan_json),
        "source_repair_manifest": str(source_repair_manifest),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "command_rows_jsonl": str(output_dir / COMMAND_ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "source_failure_taxonomy": list(source_failure_taxonomy),
        "expanded_failure_labels": list(expanded_labels),
        "commands": [],
        "execution_started": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "materializer_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_judge": True,
        "requires_result_judge": True,
        "requires_delta_gate": True,
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
        "runtime_environment": str(runtime_environment),
        "native_runtime_allowed": bool(native_runtime_allowed),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "materializer writes registered model command plans only",
        "artifact_name_policy": {
            "short_output_names": True,
            "raw_ids_in_paths": False,
            "path_segment_limit": 32,
            "output_files": [MANIFEST_NAME, ROWS_NAME, COMMAND_ROWS_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "command_args": list(command_args),
    }
    if docker_path_mapping is not None:
        manifest["docker_path_mapping"] = dict(docker_path_mapping)
    write_jsonl(output_dir / ROWS_NAME, [])
    write_jsonl(output_dir / COMMAND_ROWS_NAME, [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(progress_path, "finish", command_plan_status=status, failed_checks=list(failed_checks))
    return manifest


def materialize_minus_delta_model_candidate_route(
    *,
    router_plan_json: Path,
    source_repair_manifest: Path,
    output_dir: Path,
    registry: ModelRegistry | None = None,
    candidate_id: str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    command_args: Sequence[str] | None = None,
    runtime_environment: str = "docker",
    native_runtime_allowed: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", router_plan_json=str(router_plan_json), source_repair_manifest=str(source_repair_manifest))

    router_manifest = read_json(router_plan_json)
    router_row = load_router_row(router_manifest, router_plan_json)
    selected_route = str(router_row.get("selected_route") or router_manifest.get("selected_route") or "")
    resolved_candidate_id = str(
        candidate_id
        or router_row.get("candidate_id")
        or router_manifest.get("candidate_id")
        or "minus_delta_model_candidate"
    )
    source_failures = stable_unique(
        as_string_list(router_manifest.get("failure_taxonomy")) + as_string_list(router_row.get("failure_taxonomy"))
    )
    expanded_labels = expanded_failure_labels(router_manifest, router_row)
    progress(progress_path, "router_loaded", candidate_id=resolved_candidate_id, selected_route=selected_route)
    if selected_route != ROUTE:
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_route_not_selected",
            failed_checks=["router_selected_route_not_model_candidate_router"],
            command_args=command_args or [],
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
            source_failure_taxonomy=source_failures,
            expanded_labels=expanded_labels,
        )

    source_audio_path = source_audio_path_from_evidence(
        router_plan_json=router_plan_json,
        source_repair_manifest=source_repair_manifest,
        router_manifest=router_manifest,
        router_row=router_row,
    )
    if source_audio_path is None:
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_evidence",
            failed_checks=["source_audio_path_missing"],
            command_args=command_args or [],
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
            source_failure_taxonomy=source_failures,
            expanded_labels=expanded_labels,
        )
    if not source_audio_path.exists():
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_evidence",
            failed_checks=["source_audio_file_missing"],
            command_args=command_args or [],
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
            source_failure_taxonomy=source_failures,
            expanded_labels=expanded_labels,
        )

    docker_source_audio_path = docker_visible_path(source_audio_path)
    docker_output_dir = docker_visible_path(output_dir)
    docker_mapping = docker_path_mapping_payload(
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        docker_source_audio_path=docker_source_audio_path,
        docker_output_dir=docker_output_dir,
    )
    if not docker_source_audio_path or not docker_output_dir:
        progress(progress_path, "docker_path_mapping_failed", docker_path_mapping=docker_mapping)
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_docker_path_mapping",
            failed_checks=["docker_path_mapping_missing"],
            command_args=command_args or [],
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
            source_failure_taxonomy=source_failures,
            expanded_labels=expanded_labels,
            docker_path_mapping=docker_mapping,
        )
    progress(progress_path, "docker_path_mapping_ready", docker_path_mapping=docker_mapping)

    registry = registry or default_model_registry()
    suggestions = registry.suggest_next_candidates(
        stem_names=("instrumental", "minus"),
        failed_labels=expanded_labels,
        evaluated_labels=evaluated_labels(router_manifest, router_row),
        include_unavailable=False,
        limit=max_candidates,
    )
    progress(
        progress_path,
        "registry_suggestions_loaded",
        expanded_failure_labels=expanded_labels,
        suggestion_count=len(suggestions),
    )
    if not suggestions:
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_no_model_candidates",
            failed_checks=["available_minus_model_candidates_missing"],
            command_args=command_args or [],
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
            source_failure_taxonomy=source_failures,
            expanded_labels=expanded_labels,
        )

    rows: list[dict[str, Any]] = []
    for index, suggestion in enumerate(suggestions, start=1):
        candidate_alias = f"m{index:03d}"
        route_output_dir = output_dir / MODEL_OUTPUT_DIR_NAME / candidate_alias
        docker_route_output_dir = docker_visible_path(route_output_dir)
        if not docker_route_output_dir:
            raise RuntimeError(f"internal_docker_output_mapping_failed:{route_output_dir}")
        command = [
            "python",
            TOOL,
            docker_source_audio_path,
            docker_route_output_dir,
            "--model-id",
            suggestion.model_id,
            "--device",
            "cuda",
            "--target-stem",
            "minus",
        ]
        rows.append(
            {
                "schema": "minus_delta_model_candidate_command_row.v1",
                "route": ROUTE,
                "tool": TOOL,
                "command_status": "ready",
                "candidate_id": candidate_alias,
                "source_candidate_id": resolved_candidate_id,
                "model_id": suggestion.model_id,
                "display_name": suggestion.display_name,
                "family": suggestion.family,
                "target_stem": "minus",
                "target_stems": list(suggestion.target_stems),
                "output_stems": list(suggestion.output_stems),
                "availability": suggestion_availability_dict(suggestion),
                "reason": suggestion.reason,
                "router_plan_json": str(router_plan_json),
                "source_repair_manifest": str(source_repair_manifest),
                "source_audio_path": str(source_audio_path),
                "output_dir": str(route_output_dir),
                "docker_source_audio_path": docker_source_audio_path,
                "docker_output_dir": docker_route_output_dir,
                "docker_path_mapping": docker_mapping,
                "docker_command": command,
                "expected_acceleration_mode": "gpu_registered_model_inference",
                "runtime_environment": str(runtime_environment),
                "native_runtime_allowed": False,
                "materializer_native_runtime_allowed": bool(native_runtime_allowed),
                "requires_post_judge": True,
                "requires_result_judge": True,
                "requires_delta_gate": True,
                "delivery_allowed": False,
                "can_publish_as_result": False,
            }
        )

    manifest = {
        "schema": SCHEMA,
        "mode": "minus_delta_model_candidate_materializer",
        "command_plan_status": "ready",
        "candidate_id": resolved_candidate_id,
        "stem_type": "minus",
        "selected_route": selected_route,
        "selected_next_action": str(router_row.get("next_action") or router_manifest.get("selected_next_action") or ""),
        "ready_command_count": len(rows),
        "blocked_command_count": 0,
        "failed_checks": [],
        "router_plan_json": str(router_plan_json),
        "source_repair_manifest": str(source_repair_manifest),
        "source_audio_path": str(source_audio_path),
        "docker_source_audio_path": docker_source_audio_path,
        "docker_output_dir": docker_output_dir,
        "docker_path_mapping": docker_mapping,
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "command_rows_jsonl": str(output_dir / COMMAND_ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "source_failure_taxonomy": source_failures,
        "expanded_failure_labels": expanded_labels,
        "commands": rows,
        "execution_started": False,
        "delivery_allowed": False,
        "delivery_status": "command_plan_only",
        "presentation_status": "materializer_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_judge": True,
        "requires_result_judge": True,
        "requires_delta_gate": True,
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
        "runtime_environment": str(runtime_environment),
        "native_runtime_allowed": bool(native_runtime_allowed),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "materializer writes registered model command plans only",
        "artifact_name_policy": {
            "short_output_names": True,
            "raw_ids_in_paths": False,
            "path_segment_limit": 32,
            "output_files": [MANIFEST_NAME, ROWS_NAME, COMMAND_ROWS_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "command_args": list(command_args or []),
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_jsonl(output_dir / COMMAND_ROWS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(
        progress_path,
        "finish",
        command_plan_status="ready",
        ready_command_count=len(rows),
        manifest_path=str(output_dir / MANIFEST_NAME),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-plan-json", required=True, type=Path)
    parser.add_argument("--source-repair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    parser.add_argument("--short-output-names", action="store_true")
    parser.add_argument(
        "--allow-native-runtime",
        action="store_true",
        help="Allow native Mac metadata planning only; generated model commands remain Docker/GPU commands.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = cli_runtime_contract(allow_native_runtime=bool(args.allow_native_runtime))
    require_docker_runtime(
        tool_name="materialize_minus_delta_model_candidate_route.py",
        entrypoint_path="/app/tools/materialize_minus_delta_model_candidate_route.py",
        allow_native_macos_diagnostic=bool(args.allow_native_runtime),
    )
    command_args = list(argv if argv is not None else sys.argv[1:])
    manifest = materialize_minus_delta_model_candidate_route(
        router_plan_json=args.router_plan_json,
        source_repair_manifest=args.source_repair_manifest,
        output_dir=args.output_dir,
        registry=default_model_registry(runtime_root=args.runtime_root),
        candidate_id=args.candidate_id,
        max_candidates=args.max_candidates,
        command_args=command_args,
        runtime_environment=str(runtime["runtime_environment"]),
        native_runtime_allowed=bool(runtime["native_runtime_allowed"]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "command_plan_status": manifest["command_plan_status"],
                "ready_command_count": manifest["ready_command_count"],
                "blocked_command_count": manifest["blocked_command_count"],
                "delivery_allowed": manifest["delivery_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if manifest.get("command_plan_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
