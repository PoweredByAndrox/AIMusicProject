#!/usr/bin/env python3
"""Materialize Docker commands for documented Mac promising repair routes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.docker_runtime import require_docker_runtime  # noqa: E402


START_TIME = time.monotonic()
MANIFEST_NAME = "cmds.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
LEAK_DIR_NAME = "leak"
LEAK_ROWS_NAME = "rows.jsonl"
MISSING_PATH = Path("__missing_path__")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    return path


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_command_materialization",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "command_materializer_reads_metadata_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "command_materializer_no_training_started",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def canonical_path(value: Any, base_dir: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return MISSING_PATH
    path = Path(text)
    if path.is_absolute():
        return path
    return base_dir / path


def path_is_set(path: Path) -> bool:
    return str(path) != str(MISSING_PATH)


def source_manifest_path(readiness_summary_json: Path, plan_dir: Path) -> Path:
    summary = read_json(readiness_summary_json)
    return canonical_path(summary.get("manifest"), plan_dir)


def guard_regions_path(source_manifest: dict[str, Any], source_manifest_json: Path) -> Path:
    guard = source_manifest.get("guard") if isinstance(source_manifest.get("guard"), dict) else {}
    for key in ("regions_jsonl", "score_regions_jsonl"):
        path = canonical_path(guard.get(key), source_manifest_json.parent)
        if path_is_set(path):
            return path
    score_path = canonical_path(guard.get("score_path"), source_manifest_json.parent)
    if path_is_set(score_path):
        return score_path.parent / "regions.jsonl"
    return source_manifest_json.parent / "guard" / "score" / "regions.jsonl"


def candidate_paths(source_manifest: dict[str, Any], source_manifest_json: Path) -> dict[str, Path]:
    candidate = source_manifest.get("candidate") if isinstance(source_manifest.get("candidate"), dict) else {}
    return {
        "raw_ref": canonical_path(candidate.get("raw_ref_path"), source_manifest_json.parent),
        "minus": canonical_path(candidate.get("instrumental_path"), source_manifest_json.parent),
        "vocals": canonical_path(candidate.get("vocals_path"), source_manifest_json.parent),
    }


def leakage_judge_rows(
    *,
    guard_regions_jsonl: Path,
    candidate_id: str,
    raw_ref_path: Path,
    minus_path: Path,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for region in iter_jsonl(guard_regions_jsonl):
        start = float(region.get("start") or region.get("start_sec") or 0.0)
        end = float(region.get("end") or region.get("end_sec") or start)
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue
        score = float(region.get("score") or 0.0)
        windows.append(
            {
                "start_sec": round(start, 6),
                "duration_sec": round(duration, 6),
                "blocker": True,
                "issues": ["vocal_leakage_in_minus"],
                "minus_blockers": ["vocal_leakage_in_minus"],
                "vocal_leakage_in_minus_score": round(score, 6),
                "severity_score": round(score, 6),
                "source_guard_fail_mode": str(region.get("fail_mode") or ""),
                "source_guard_projection_rel_db": region.get("projection_rel_db"),
            }
        )
    if not windows:
        return []
    return [
        {
            "schema": "windowed_minus_judge_row.v1",
            "candidate_id": candidate_id,
            "candidate_label": candidate_id,
            "source_audio_path": str(raw_ref_path),
            "minus_path": str(minus_path),
            "instrumental_path": str(minus_path),
            "judge_source": "mac_pitch_guard_regions",
            "window_metrics": windows,
        }
    ]


def materialize_route_row(
    *,
    route: dict[str, Any],
    queue_item_id: str,
    source_manifest_json: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    source_manifest = read_json(source_manifest_json)
    paths = candidate_paths(source_manifest, source_manifest_json)
    missing = [name for name, path in paths.items() if not path_is_set(path) or not path.exists()]
    route_name = str(route.get("route") or "")
    if route_name == "judge_guided_minus_air_detail_repair":
        return (
            {
                "schema": "mac_promising_repair_command_row.v1",
                "route": route_name,
                "tool": str(route.get("tool") or ""),
                "command_status": "blocked_missing_windowed_detail_judge",
                "missing_inputs": ["self_judge_jsonl"],
                "source_manifest_json": str(source_manifest_json),
                "delivery_allowed": False,
                "can_publish_as_result": False,
                "docker_command": [],
            },
            ["detail_route_windowed_judge_missing"],
        )
    if route_name != "judge_guided_minus_vocal_leakage_repair":
        return (
            {
                "schema": "mac_promising_repair_command_row.v1",
                "route": route_name,
                "tool": str(route.get("tool") or ""),
                "command_status": "blocked_unsupported_route",
                "missing_inputs": ["supported_route_mapping"],
                "source_manifest_json": str(source_manifest_json),
                "delivery_allowed": False,
                "can_publish_as_result": False,
                "docker_command": [],
            },
            [f"unsupported_route:{route_name}"],
        )

    guard_path = guard_regions_path(source_manifest, source_manifest_json)
    if not path_is_set(guard_path) or not guard_path.exists():
        missing.append("guard_regions_jsonl")
    if missing:
        return (
            {
                "schema": "mac_promising_repair_command_row.v1",
                "route": route_name,
                "tool": str(route.get("tool") or ""),
                "command_status": "blocked_missing_inputs",
                "missing_inputs": stable_unique(missing),
                "source_manifest_json": str(source_manifest_json),
                "delivery_allowed": False,
                "can_publish_as_result": False,
                "docker_command": [],
            },
            [f"{route_name}_missing_inputs"],
        )

    leak_rows_path = output_dir / LEAK_DIR_NAME / LEAK_ROWS_NAME
    leak_rows = leakage_judge_rows(
        guard_regions_jsonl=guard_path,
        candidate_id=queue_item_id,
        raw_ref_path=paths["raw_ref"],
        minus_path=paths["minus"],
    )
    write_jsonl(leak_rows_path, leak_rows)
    if not leak_rows:
        return (
            {
                "schema": "mac_promising_repair_command_row.v1",
                "route": route_name,
                "tool": str(route.get("tool") or ""),
                "command_status": "blocked_no_guard_leakage_regions",
                "missing_inputs": ["guard_leakage_regions"],
                "source_manifest_json": str(source_manifest_json),
                "generated_judge_jsonl": str(leak_rows_path),
                "delivery_allowed": False,
                "can_publish_as_result": False,
                "docker_command": [],
            },
            ["leakage_route_guard_regions_missing"],
        )

    route_output_dir = output_dir / "lkg"
    command = [
        "python",
        str(route.get("tool") or "tools/apply_judge_guided_minus_vocal_leakage_repair.py"),
        str(paths["raw_ref"]),
        str(paths["minus"]),
        str(paths["vocals"]),
        str(leak_rows_path),
        str(route_output_dir),
        "--candidate-id",
        queue_item_id,
        "--track-id",
        queue_item_id,
        "--short-output-names",
    ]
    return (
        {
            "schema": "mac_promising_repair_command_row.v1",
            "route": route_name,
            "tool": str(route.get("tool") or ""),
            "command_status": "ready",
            "missing_inputs": [],
            "source_manifest_json": str(source_manifest_json),
            "raw_ref_path": str(paths["raw_ref"]),
            "base_minus_path": str(paths["minus"]),
            "primary_vocals_path": str(paths["vocals"]),
            "source_guard_regions_jsonl": str(guard_path),
            "generated_judge_jsonl": str(leak_rows_path),
            "output_dir": str(route_output_dir),
            "docker_command": command,
            "expected_acceleration_mode": "external_only_cpu_dsp_repair",
            "requires_post_judge": True,
            "delivery_allowed": False,
            "can_publish_as_result": False,
        },
        [],
    )


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Mac Promising Repair Commands",
        "",
        f"- command_plan_status: `{manifest['command_plan_status']}`",
        f"- ready_command_count: `{manifest['ready_command_count']}`",
        f"- blocked_command_count: `{manifest['blocked_command_count']}`",
        f"- delivery_allowed: `{str(manifest['delivery_allowed']).lower()}`",
        "",
        "## Routes",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['route']}` `{row['command_status']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def materialize_mac_promising_repair_commands(
    *,
    plan_json: Path,
    output_dir: Path,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", plan_json=str(plan_json))

    plan = read_json(plan_json)
    failed: list[str] = []
    if plan.get("schema") != "full_pack_loop_iteration_plan.v1":
        failed.append("plan_schema_unexpected")
    if plan.get("delivery_allowed") is not False:
        failed.append("plan_delivery_allowed_not_false")
    if plan.get("failed_checks") not in ([], None):
        failed.append("plan_failed_checks_not_empty")
    rows_path = canonical_path(plan.get("rows_jsonl"), plan_json.parent)
    plan_rows = list(iter_jsonl(rows_path)) if path_is_set(rows_path) and rows_path.exists() else []
    if not plan_rows:
        failed.append("plan_rows_missing")

    command_rows: list[dict[str, Any]] = []
    for plan_row in plan_rows:
        if str(plan_row.get("route_intent") or "") != "mac_promising_blocked_handoff":
            continue
        readiness_path = canonical_path(plan_row.get("readiness_summary_json"), plan_json.parent)
        if not path_is_set(readiness_path) or not readiness_path.exists():
            failed.append("readiness_summary_missing")
            continue
        source_manifest_json = source_manifest_path(readiness_path, readiness_path.parent)
        if not path_is_set(source_manifest_json) or not source_manifest_json.exists():
            failed.append("source_manifest_missing")
            continue
        queue_item_id = str(plan_row.get("queue_item_id") or "mph001")
        routes = [route for route in plan_row.get("recommended_routes") or [] if isinstance(route, dict)]
        for route in routes:
            row, row_failed = materialize_route_row(
                route=route,
                queue_item_id=queue_item_id,
                source_manifest_json=source_manifest_json,
                output_dir=output_dir,
            )
            command_rows.append(row)
            failed.extend(row_failed)

    failed = stable_unique(failed)
    ready_count = sum(1 for row in command_rows if row.get("command_status") == "ready")
    blocked_count = len(command_rows) - ready_count
    if failed and ready_count:
        status = "partial_ready_missing_inputs"
    elif failed:
        status = "blocked_missing_inputs"
    elif ready_count:
        status = "ready"
    else:
        status = "blocked_no_commands"
    manifest = {
        "schema": "mac_promising_repair_command_plan.v1",
        "command_plan_status": status,
        "plan_json": str(plan_json),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "ready_command_count": ready_count,
        "blocked_command_count": blocked_count,
        "failed_checks": failed,
        "delivery_allowed": False,
        "release_ready": False,
        "can_publish_as_result": False,
        "execution_started": False,
        "training_started": False,
        "fine_tune_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "presentation_status": "command_plan_not_for_listen_pack",
        "command_args": command_args or [],
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    write_jsonl(output_dir / ROWS_NAME, command_rows)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, command_rows), encoding="utf-8")
    progress(progress_path, "finish", command_plan_status=status, ready_command_count=ready_count, blocked_command_count=blocked_count, failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="materialize_mac_promising_repair_commands.py",
        entrypoint_path="/app/tools/materialize_mac_promising_repair_commands.py",
    )
    args = parse_args(argv)
    manifest = materialize_mac_promising_repair_commands(
        plan_json=args.plan_json,
        output_dir=args.output_dir,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "command_plan_status": manifest["command_plan_status"],
                "ready_command_count": manifest["ready_command_count"],
                "blocked_command_count": manifest["blocked_command_count"],
                "failed_checks": manifest["failed_checks"],
                "delivery_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["command_plan_status"] == "ready" and not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
