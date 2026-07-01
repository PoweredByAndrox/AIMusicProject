#!/usr/bin/env python3
"""Materialize router-selected minus leakage repair commands from delta evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from docker_runtime import cli_runtime_contract, require_docker_runtime


SCHEMA = "minus_delta_leakage_repair_command_plan.v1"
ROUTE = "judge_guided_minus_vocal_leakage_repair"
TOOL = "tools/apply_judge_guided_minus_vocal_leakage_repair.py"
MANIFEST_NAME = "plan.json"
ROWS_NAME = "rows.jsonl"
COMMAND_ROWS_NAME = "cmds.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
OUTPUT_DIR_NAME = "lkg"

UNSAFE_LEAKAGE_ISSUES = {
    "vocal_projection_proxy",
    "vocal_stem_corr_proxy",
    "full_mix_like_risk_proxy",
    "vocal_leakage_in_minus",
}
ISSUE_SEVERITY_KEYS = {
    "vocal_projection_proxy": "vocal_projection",
    "vocal_stem_corr_proxy": "vocal_stem_corr",
    "full_mix_like_risk_proxy": "full_mix_like_risk",
    "vocal_leakage_in_minus": "vocal_leakage_in_minus",
}
REPAIR_TRANSITIONS = {"persisting_blocker", "new_blocker"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def progress(path: Path, phase: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"phase": phase, "time_unix": round(time.time(), 6), **payload}
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


def canonical_path(value: Any, base_dir: Path) -> Path:
    text = str(value or "")
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def maybe_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    return canonical_path(value, base_dir)


def nested_get(payload: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_path(payload: dict[str, Any], base_dir: Path, candidates: Sequence[Sequence[str]]) -> Path | None:
    for keys in candidates:
        path = maybe_path(nested_get(payload, keys), base_dir)
        if path is not None:
            return path
    return None


def row_issues(row: dict[str, Any], key: str = "issues") -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        return []
    return stable_unique(value)


def row_has_leakage_issue(row: dict[str, Any]) -> bool:
    return bool(set(row_issues(row, "after_issues")) & UNSAFE_LEAKAGE_ISSUES)


def window_bounds(row: dict[str, Any]) -> tuple[float, float] | None:
    window_sec = row.get("window_sec")
    if (
        isinstance(window_sec, list)
        and len(window_sec) == 2
        and all(isinstance(value, (int, float)) for value in window_sec)
    ):
        start = float(window_sec[0])
        end = float(window_sec[1])
        if end > start:
            return start, end - start
    start = row.get("start_sec")
    duration = row.get("duration_sec")
    if isinstance(start, (int, float)) and isinstance(duration, (int, float)) and float(duration) > 0:
        return float(start), float(duration)
    return None


def leakage_score(row: dict[str, Any], after_issues: Sequence[str]) -> float:
    severities = row.get("issue_severities")
    scores: list[float] = []
    if isinstance(severities, dict):
        for issue in after_issues:
            key = ISSUE_SEVERITY_KEYS.get(str(issue))
            value = severities.get(key) if key else None
            if isinstance(value, (int, float)):
                scores.append(float(value))
    direct = row.get("vocal_leakage_in_minus_score")
    if isinstance(direct, (int, float)):
        scores.append(float(direct))
    if not scores and isinstance(row.get("severity_score"), (int, float)):
        scores.append(float(row["severity_score"]))
    return round(max(scores) if scores else 0.000001, 6)


def selected_delta_rows(delta_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in delta_rows:
        if str(row.get("transition") or "") not in REPAIR_TRANSITIONS:
            continue
        if row.get("after_blocker") is not True:
            continue
        if not row_has_leakage_issue(row):
            continue
        result.append(row)
    return result


def normalize_window(delta_row: dict[str, Any], after_row: dict[str, Any]) -> dict[str, Any] | None:
    bounds = window_bounds(after_row)
    if bounds is None:
        return None
    start_sec, duration_sec = bounds
    issues = stable_unique(row_issues(after_row) + row_issues(delta_row, "after_issues"))
    after_issues = row_issues(delta_row, "after_issues")
    converted = dict(after_row)
    converted["schema"] = "windowed_minus_judge_row.v1"
    converted["segment_id"] = str(delta_row.get("segment_id") or after_row.get("segment_id") or "")
    converted["window_sec"] = [round(start_sec, 6), round(start_sec + duration_sec, 6)]
    converted["start_sec"] = round(start_sec, 6)
    converted["duration_sec"] = round(duration_sec, 6)
    converted["issues"] = issues
    converted["minus_blockers"] = ["vocal_leakage_in_minus"]
    converted["source_delta_transition"] = str(delta_row.get("transition") or "")
    converted["source_delta_after_issues"] = after_issues
    converted["source_delta_before_issues"] = row_issues(delta_row, "before_issues")
    converted["target_window_source"] = "minus_repair_delta_gate_after_self_judge"
    converted["vocal_leakage_in_minus_score"] = leakage_score(after_row, after_issues)
    return converted


def command_status_for_missing(missing_inputs: Sequence[str]) -> str:
    return "blocked_missing_inputs" if missing_inputs else "ready"


def summary_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Minus Delta Leakage Repair Materializer",
        "",
        f"- command_plan_status: `{manifest['command_plan_status']}`",
        f"- selected_route: `{manifest.get('selected_route') or ''}`",
        f"- selected_window_count: `{manifest['selected_window_count']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        "",
        "## Failed Checks",
        "",
    ]
    failed_checks = manifest.get("failed_checks") or []
    lines.extend(f"- `{item}`" for item in failed_checks) if failed_checks else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def load_router_row(router_manifest: dict[str, Any], router_path: Path) -> dict[str, Any]:
    rows = router_manifest.get("rows")
    if isinstance(rows, list) and rows:
        return dict(rows[0])
    rows_path = maybe_path(router_manifest.get("rows_jsonl"), router_path.parent)
    if rows_path is not None and rows_path.exists():
        rows_from_file = list(iter_jsonl(rows_path))
        if rows_from_file:
            return dict(rows_from_file[0])
    return {}


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
    blocked_repeat_routes: Sequence[str] = (),
    runtime_environment: str = "docker",
    native_runtime_allowed: bool = False,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA,
        "mode": "minus_delta_leakage_repair_materializer",
        "command_plan_status": status,
        "candidate_id": candidate_id,
        "stem_type": "minus",
        "selected_route": selected_route,
        "selected_window_count": 0,
        "ready_command_count": 0,
        "blocked_command_count": 1,
        "failed_checks": list(failed_checks),
        "blocked_repeat_routes": list(blocked_repeat_routes),
        "router_plan_json": str(router_plan_json),
        "source_repair_manifest": str(source_repair_manifest),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "command_rows_jsonl": str(output_dir / COMMAND_ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "execution_started": False,
        "delivery_allowed": False,
        "delivery_status": "blocked",
        "presentation_status": "materializer_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_judge": True,
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
        "runtime_environment": str(runtime_environment),
        "native_runtime_allowed": bool(native_runtime_allowed),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "materializer reads existing router, delta, and self-judge JSON evidence only",
        "artifact_name_policy": {
            "short_output_names": True,
            "raw_ids_in_paths": False,
            "path_segment_limit": 32,
            "output_files": [MANIFEST_NAME, ROWS_NAME, COMMAND_ROWS_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "command_args": list(command_args),
    }
    write_jsonl(output_dir / ROWS_NAME, [])
    write_jsonl(output_dir / COMMAND_ROWS_NAME, [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(progress_path, "finish", command_plan_status=status, failed_checks=list(failed_checks))
    return manifest


def materialize_minus_delta_leakage_repair(
    *,
    router_plan_json: Path,
    source_repair_manifest: Path,
    output_dir: Path,
    candidate_id: str | None = None,
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
        or "minus_delta_leakage_candidate"
    )
    blocked_repeat_routes = stable_unique(
        list(router_row.get("blocked_repeat_routes") or []) + list(router_manifest.get("blocked_repeat_routes") or [])
    )
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
            failed_checks=["router_selected_route_not_leakage_repair"],
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )

    delta_gate_path = maybe_path(router_row.get("delta_gate_manifest") or router_manifest.get("delta_gate_manifest"), router_plan_json.parent)
    if delta_gate_path is None or not delta_gate_path.exists():
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_evidence",
            failed_checks=["delta_gate_manifest_missing"],
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )
    delta_gate = read_json(delta_gate_path)
    delta_rows_path = maybe_path(delta_gate.get("rows_jsonl"), delta_gate_path.parent)
    after_manifest_path = maybe_path(delta_gate.get("after_self_judge_manifest"), delta_gate_path.parent)
    if delta_rows_path is None or not delta_rows_path.exists() or after_manifest_path is None or not after_manifest_path.exists():
        missing = []
        if delta_rows_path is None or not delta_rows_path.exists():
            missing.append("delta_rows_jsonl_missing")
        if after_manifest_path is None or not after_manifest_path.exists():
            missing.append("after_self_judge_manifest_missing")
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_evidence",
            failed_checks=missing,
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )

    delta_rows = list(iter_jsonl(delta_rows_path))
    selected_deltas = selected_delta_rows(delta_rows)
    progress(progress_path, "delta_loaded", delta_row_count=len(delta_rows), selected_delta_count=len(selected_deltas))
    if not selected_deltas:
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_no_persistent_unsafe_delta_windows",
            failed_checks=["persistent_unsafe_delta_windows_missing"],
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )

    after_manifest = read_json(after_manifest_path)
    after_rows_path = maybe_path(after_manifest.get("segments_jsonl"), after_manifest_path.parent)
    if after_rows_path is None or not after_rows_path.exists():
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_evidence",
            failed_checks=["after_segments_jsonl_missing"],
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )
    after_rows = list(iter_jsonl(after_rows_path))
    after_by_segment = {str(row.get("segment_id") or ""): row for row in after_rows}
    windows: list[dict[str, Any]] = []
    missing_segments: list[str] = []
    missing_timing: list[str] = []
    for delta_row in selected_deltas:
        segment_id = str(delta_row.get("segment_id") or "")
        after_row = after_by_segment.get(segment_id)
        if after_row is None:
            missing_segments.append(segment_id)
            continue
        window = normalize_window(delta_row, after_row)
        if window is None:
            missing_timing.append(segment_id)
            continue
        windows.append(window)
    if missing_segments or missing_timing:
        failed = []
        if missing_segments:
            failed.append("after_self_judge_segments_missing")
        if missing_timing:
            failed.append("after_self_judge_window_timing_missing")
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status="blocked_missing_evidence",
            failed_checks=failed,
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )

    source_repair = read_json(source_repair_manifest)
    source_base = source_repair_manifest.parent
    raw_ref_path = first_path(
        source_repair,
        source_base,
        (("raw_audio_path",), ("raw_ref_path",), ("source_audio_path",), ("candidate", "raw_ref_path")),
    ) or maybe_path(after_manifest.get("source_audio_path"), after_manifest_path.parent)
    minus_path = first_path(
        source_repair,
        source_base,
        (
            ("rendered_minus_candidate_path",),
            ("candidate_audio_path",),
            ("artifacts", "minus_judge_guided_leakage_repair_flac"),
            ("artifacts", "minus_judge_guided_air_detail_repair_flac"),
            ("base_minus_path",),
            ("candidate", "instrumental_path"),
        ),
    ) or first_path(after_manifest, after_manifest_path.parent, (("candidate", "instrumental_path"),))
    vocal_path = first_path(
        source_repair,
        source_base,
        (("vocal_guide_path",), ("primary_vocals_path",), ("candidate", "vocal_path"), ("candidate", "vocals_path")),
    ) or first_path(after_manifest, after_manifest_path.parent, (("candidate", "vocal_path"),))

    paths = {"raw_ref_path": raw_ref_path, "minus_path": minus_path, "primary_vocals_path": vocal_path}
    missing_inputs = [name for name, path in paths.items() if path is None or not path.exists()]
    if missing_inputs:
        return blocked_manifest(
            output_dir=output_dir,
            progress_path=progress_path,
            router_plan_json=router_plan_json,
            source_repair_manifest=source_repair_manifest,
            candidate_id=resolved_candidate_id,
            selected_route=selected_route,
            status=command_status_for_missing(missing_inputs),
            failed_checks=missing_inputs,
            command_args=command_args or [],
            blocked_repeat_routes=blocked_repeat_routes,
            runtime_environment=runtime_environment,
            native_runtime_allowed=native_runtime_allowed,
        )
    assert raw_ref_path is not None
    assert minus_path is not None
    assert vocal_path is not None

    judge_row = {
        "schema": "windowed_minus_judge_row.v1",
        "candidate_id": resolved_candidate_id,
        "candidate_label": resolved_candidate_id,
        "source_audio_path": str(raw_ref_path),
        "minus_path": str(minus_path),
        "instrumental_path": str(minus_path),
        "primary_vocals_path": str(vocal_path),
        "judge_source": "minus_repair_delta_gate_after_self_judge",
        "delta_gate_manifest": str(delta_gate_path),
        "after_self_judge_manifest": str(after_manifest_path),
        "source_delta_rows_jsonl": str(delta_rows_path),
        "source_self_judge_rows_jsonl": str(after_rows_path),
        "window_metrics": windows,
    }
    judge_rows_path = output_dir / ROWS_NAME
    command_rows_path = output_dir / COMMAND_ROWS_NAME
    route_output_dir = output_dir / OUTPUT_DIR_NAME
    command = [
        "python",
        TOOL,
        str(raw_ref_path),
        str(minus_path),
        str(vocal_path),
        str(judge_rows_path),
        str(route_output_dir),
        "--candidate-id",
        resolved_candidate_id,
        "--track-id",
        resolved_candidate_id,
        "--short-output-names",
    ]
    native_command = [*command, "--allow-native-runtime"] if native_runtime_allowed else []
    command_row = {
        "schema": "minus_delta_leakage_repair_command_row.v1",
        "route": ROUTE,
        "tool": TOOL,
        "command_status": "ready",
        "candidate_id": resolved_candidate_id,
        "missing_inputs": [],
        "router_plan_json": str(router_plan_json),
        "source_repair_manifest": str(source_repair_manifest),
        "delta_gate_manifest": str(delta_gate_path),
        "after_self_judge_manifest": str(after_manifest_path),
        "source_delta_rows_jsonl": str(delta_rows_path),
        "source_self_judge_rows_jsonl": str(after_rows_path),
        "generated_judge_jsonl": str(judge_rows_path),
        "raw_ref_path": str(raw_ref_path),
        "base_minus_path": str(minus_path),
        "primary_vocals_path": str(vocal_path),
        "output_dir": str(route_output_dir),
        "docker_command": command,
        "native_command": native_command,
        "selected_window_count": len(windows),
        "expected_acceleration_mode": "external_only_cpu_dsp_repair",
        "runtime_environment": str(runtime_environment),
        "native_runtime_allowed": bool(native_runtime_allowed),
        "requires_post_judge": True,
        "delivery_allowed": False,
        "can_publish_as_result": False,
    }
    manifest = {
        "schema": SCHEMA,
        "mode": "minus_delta_leakage_repair_materializer",
        "command_plan_status": "ready",
        "candidate_id": resolved_candidate_id,
        "stem_type": "minus",
        "selected_route": selected_route,
        "selected_next_action": str(router_row.get("next_action") or router_manifest.get("selected_next_action") or ""),
        "selected_window_count": len(windows),
        "ready_command_count": 1,
        "blocked_command_count": 0,
        "failed_checks": [],
        "blocked_repeat_routes": blocked_repeat_routes,
        "router_plan_json": str(router_plan_json),
        "source_repair_manifest": str(source_repair_manifest),
        "delta_gate_manifest": str(delta_gate_path),
        "after_self_judge_manifest": str(after_manifest_path),
        "rows_jsonl": str(judge_rows_path),
        "command_rows_jsonl": str(command_rows_path),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "commands": [command_row],
        "execution_started": False,
        "delivery_allowed": False,
        "delivery_status": "command_plan_only",
        "presentation_status": "materializer_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "requires_post_judge": True,
        "requires_delta_gate": True,
        "expected_acceleration_mode": "external_only_cpu_metadata_planning",
        "runtime_environment": str(runtime_environment),
        "native_runtime_allowed": bool(native_runtime_allowed),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "materializer reads existing router, delta, and self-judge JSON evidence only",
        "artifact_name_policy": {
            "short_output_names": True,
            "raw_ids_in_paths": False,
            "path_segment_limit": 32,
            "output_files": [MANIFEST_NAME, ROWS_NAME, COMMAND_ROWS_NAME, SUMMARY_NAME, PROGRESS_NAME],
        },
        "command_args": list(command_args or []),
    }
    write_jsonl(judge_rows_path, [judge_row])
    write_jsonl(command_rows_path, [command_row])
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(progress_path, "finish", command_plan_status="ready", selected_window_count=len(windows), manifest_path=str(output_dir / MANIFEST_NAME))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-plan-json", required=True, type=Path)
    parser.add_argument("--source-repair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--short-output-names", action="store_true")
    parser.add_argument(
        "--allow-native-runtime",
        action="store_true",
        help="Explicitly allow native Mac CPU-DSP diagnostic execution outside Docker.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = cli_runtime_contract(allow_native_runtime=bool(args.allow_native_runtime))
    require_docker_runtime(
        tool_name="materialize_minus_delta_leakage_repair.py",
        entrypoint_path="/app/tools/materialize_minus_delta_leakage_repair.py",
        allow_native_macos_diagnostic=bool(args.allow_native_runtime),
    )
    command_args = list(argv if argv is not None else sys.argv[1:])
    manifest = materialize_minus_delta_leakage_repair(
        router_plan_json=args.router_plan_json,
        source_repair_manifest=args.source_repair_manifest,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        command_args=command_args,
        runtime_environment=str(runtime["runtime_environment"]),
        native_runtime_allowed=bool(runtime["native_runtime_allowed"]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "command_plan_status": manifest["command_plan_status"],
                "selected_window_count": manifest["selected_window_count"],
                "ready_command_count": manifest["ready_command_count"],
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
