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
DETAIL_DIR_NAME = "dtl"
DETAIL_ROWS_NAME = "rows.jsonl"
AIR_DETAIL_OUTPUT_DIR_NAME = "air"
MISSING_PATH = Path("__missing_path__")

# Mirrors the documented issue contract consumed by
# apply_judge_guided_minus_air_detail_repair.py without importing the audio stack.
DETAIL_ISSUES = {"air_loss_proxy", "presence_loss_proxy", "low_rms_muffling_proxy"}
UNSAFE_DETAIL_ISSUES = {"vocal_projection_proxy", "vocal_stem_corr_proxy", "full_mix_like_risk_proxy"}


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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalized_path_text(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


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


def first_manifest_path(mapping: dict[str, Any], keys: Sequence[str], base_dir: Path) -> Path:
    for key in keys:
        path = canonical_path(mapping.get(key), base_dir)
        if path_is_set(path):
            return path
    return MISSING_PATH


def candidate_paths(source_manifest: dict[str, Any], source_manifest_json: Path) -> dict[str, Path]:
    candidate = source_manifest.get("candidate") if isinstance(source_manifest.get("candidate"), dict) else {}
    stem_paths = source_manifest.get("stem_paths") if isinstance(source_manifest.get("stem_paths"), dict) else {}
    stem_minus = first_manifest_path(stem_paths, ("minus", "instrumental"), source_manifest_json.parent)
    stem_vocals = first_manifest_path(stem_paths, ("vocals", "vocal"), source_manifest_json.parent)
    return {
        "raw_ref": canonical_path(candidate.get("raw_ref_path"), source_manifest_json.parent),
        "minus": stem_minus
        if path_is_set(stem_minus)
        else first_manifest_path(
            candidate,
            ("minus_path", "instrumental_path", "selected_minus_path", "normalized_minus_path"),
            source_manifest_json.parent,
        ),
        "vocals": stem_vocals
        if path_is_set(stem_vocals)
        else first_manifest_path(
            candidate,
            ("vocals_path", "vocal_path", "selected_vocals_path", "normalized_vocal_path"),
            source_manifest_json.parent,
        ),
    }


def self_judge_manifest_inputs(self_judge_manifest_json: Path) -> dict[str, Any]:
    manifest = read_json(self_judge_manifest_json)
    candidate = manifest.get("candidate") if isinstance(manifest.get("candidate"), dict) else {}
    raw_ref_path = canonical_path(manifest.get("source_audio_path"), self_judge_manifest_json.parent)
    segments_path = canonical_path(manifest.get("segments_jsonl"), self_judge_manifest_json.parent)
    minus_path = canonical_path(
        candidate.get("instrumental_path")
        or candidate.get("minus_path")
        or candidate.get("selected_minus_path")
        or manifest.get("candidate_instrumental_path"),
        self_judge_manifest_json.parent,
    )
    vocal_path = canonical_path(candidate.get("vocal_path"), self_judge_manifest_json.parent)
    candidate_label = str(candidate.get("label") or manifest.get("candidate_label") or "")
    if not candidate_label and path_is_set(minus_path):
        candidate_label = minus_path.stem
    return {
        "manifest": manifest,
        "manifest_json": self_judge_manifest_json,
        "raw_ref": raw_ref_path,
        "segments": segments_path,
        "minus": minus_path,
        "vocals": vocal_path,
        "candidate_label": candidate_label,
    }


def select_self_judge_inputs(
    *,
    self_judge_manifest_jsons: Sequence[Path],
    raw_ref_path: Path,
) -> tuple[dict[str, Any] | None, int]:
    inputs: list[dict[str, Any]] = []
    for manifest_json in self_judge_manifest_jsons:
        if manifest_json.exists():
            inputs.append(self_judge_manifest_inputs(manifest_json))
    if not inputs:
        return None, 0
    requested = normalized_path_text(raw_ref_path)
    matches = [item for item in inputs if normalized_path_text(item["raw_ref"]) == requested]
    if len(matches) == 1:
        return matches[0], 0
    if len(matches) > 1:
        return None, len(matches)
    if len(inputs) == 1:
        return inputs[0], 0
    return None, len(inputs)


def row_issues(row: dict[str, Any]) -> set[str]:
    return {str(issue) for issue in as_list(row.get("issues")) if str(issue)}


def detail_judge_rows(
    *,
    self_judge_rows_jsonl: Path,
    candidate_id: str,
    raw_ref_path: Path,
    minus_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    unsafe_detail_count = 0
    for row in iter_jsonl(self_judge_rows_jsonl):
        issues = row_issues(row)
        has_detail_issue = bool(issues & DETAIL_ISSUES)
        has_unsafe_issue = bool(issues & UNSAFE_DETAIL_ISSUES)
        if has_detail_issue and has_unsafe_issue:
            unsafe_detail_count += 1
        if not bool(row.get("blocker")) or not has_detail_issue or has_unsafe_issue:
            continue
        converted = dict(row)
        converted["schema"] = "windowed_minus_judge_row.v1"
        converted["candidate_id"] = candidate_id
        converted["candidate_label"] = str(row.get("candidate_label") or candidate_id)
        converted["source_audio_path"] = str(raw_ref_path)
        converted["minus_path"] = str(minus_path)
        converted["instrumental_path"] = str(minus_path)
        converted["judge_source"] = "instrumental_self_judge_gate"
        converted["detail_materializer_source"] = "minus_self_judge_manifest"
        rows.append(converted)
    return rows, unsafe_detail_count


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
    minus_self_judge_manifest_jsons: Sequence[Path],
) -> tuple[dict[str, Any], list[str]]:
    source_manifest = read_json(source_manifest_json)
    paths = candidate_paths(source_manifest, source_manifest_json)
    missing = [name for name, path in paths.items() if not path_is_set(path) or not path.exists()]
    route_name = str(route.get("route") or "")
    if route_name == "judge_guided_minus_air_detail_repair":
        source_missing = [name for name in ("raw_ref",) if name in missing]
        self_judge_inputs, ambiguous_count = select_self_judge_inputs(
            self_judge_manifest_jsons=minus_self_judge_manifest_jsons,
            raw_ref_path=paths["raw_ref"],
        )
        if ambiguous_count:
            return (
                {
                    "schema": "mac_promising_repair_command_row.v1",
                    "route": route_name,
                    "tool": str(route.get("tool") or ""),
                    "command_status": "blocked_ambiguous_windowed_detail_judge",
                    "missing_inputs": ["unambiguous_self_judge_manifest"],
                    "source_manifest_json": str(source_manifest_json),
                    "candidate_self_judge_manifest_count": ambiguous_count,
                    "delivery_allowed": False,
                    "can_publish_as_result": False,
                    "docker_command": [],
                },
                ["detail_route_self_judge_manifest_ambiguous"],
            )
        if self_judge_inputs is None:
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
        raw_ref_path = self_judge_inputs["raw_ref"] if path_is_set(self_judge_inputs["raw_ref"]) else paths["raw_ref"]
        detail_missing = list(source_missing)
        if not path_is_set(self_judge_inputs["raw_ref"]):
            detail_missing.append("self_judge_source_audio_path")
        elif normalized_path_text(self_judge_inputs["raw_ref"]) != normalized_path_text(paths["raw_ref"]):
            detail_missing.append("self_judge_source_audio_path_mismatch")
        for name in ("segments", "minus"):
            path = self_judge_inputs[name]
            if not path_is_set(path) or not path.exists():
                detail_missing.append(f"self_judge_{name}")
        vocal_path = self_judge_inputs["vocals"]
        if path_is_set(vocal_path) and not vocal_path.exists():
            detail_missing.append("self_judge_vocals")
        if not path_is_set(raw_ref_path) or not raw_ref_path.exists():
            detail_missing.append("raw_ref")
        if detail_missing:
            return (
                {
                    "schema": "mac_promising_repair_command_row.v1",
                    "route": route_name,
                    "tool": str(route.get("tool") or ""),
                    "command_status": "blocked_missing_windowed_detail_judge",
                    "missing_inputs": stable_unique(detail_missing),
                    "source_manifest_json": str(source_manifest_json),
                    "self_judge_manifest_json": str(self_judge_inputs["manifest_json"]),
                    "delivery_allowed": False,
                    "can_publish_as_result": False,
                    "docker_command": [],
                },
                ["detail_route_windowed_judge_missing"],
            )
        candidate_label = str(self_judge_inputs.get("candidate_label") or queue_item_id)
        detail_rows_path = output_dir / DETAIL_DIR_NAME / DETAIL_ROWS_NAME
        detail_rows, unsafe_detail_count = detail_judge_rows(
            self_judge_rows_jsonl=self_judge_inputs["segments"],
            candidate_id=candidate_label,
            raw_ref_path=raw_ref_path,
            minus_path=self_judge_inputs["minus"],
        )
        write_jsonl(detail_rows_path, detail_rows)
        if not detail_rows:
            return (
                {
                    "schema": "mac_promising_repair_command_row.v1",
                    "route": route_name,
                    "tool": str(route.get("tool") or ""),
                    "command_status": "blocked_no_safe_detail_windows",
                    "missing_inputs": ["safe_detail_windows"],
                    "source_manifest_json": str(source_manifest_json),
                    "self_judge_manifest_json": str(self_judge_inputs["manifest_json"]),
                    "source_self_judge_jsonl": str(self_judge_inputs["segments"]),
                    "generated_detail_jsonl": str(detail_rows_path),
                    "blocked_unsafe_detail_window_count": unsafe_detail_count,
                    "delivery_allowed": False,
                    "can_publish_as_result": False,
                    "docker_command": [],
                },
                ["detail_route_safe_detail_windows_missing"],
            )
        route_output_dir = output_dir / AIR_DETAIL_OUTPUT_DIR_NAME
        command = [
            "python",
            str(route.get("tool") or "tools/apply_judge_guided_minus_air_detail_repair.py"),
            str(raw_ref_path),
            str(self_judge_inputs["minus"]),
            str(detail_rows_path),
            str(route_output_dir),
            "--candidate-id",
            candidate_label,
            "--track-id",
            queue_item_id,
            "--short-output-names",
        ]
        if path_is_set(vocal_path):
            command.extend(["--vocal-guide-path", str(vocal_path)])
        return (
            {
                "schema": "mac_promising_repair_command_row.v1",
                "route": route_name,
                "tool": str(route.get("tool") or ""),
                "command_status": "ready",
                "missing_inputs": [],
                "source_manifest_json": str(source_manifest_json),
                "self_judge_manifest_json": str(self_judge_inputs["manifest_json"]),
                "source_self_judge_jsonl": str(self_judge_inputs["segments"]),
                "generated_detail_jsonl": str(detail_rows_path),
                "raw_ref_path": str(raw_ref_path),
                "base_minus_path": str(self_judge_inputs["minus"]),
                "vocal_guide_path": str(vocal_path) if path_is_set(vocal_path) else None,
                "output_dir": str(route_output_dir),
                "docker_command": command,
                "safe_detail_window_count": len(detail_rows),
                "blocked_unsafe_detail_window_count": unsafe_detail_count,
                "expected_acceleration_mode": "external_only_cpu_dsp_repair",
                "requires_post_judge": True,
                "delivery_allowed": False,
                "can_publish_as_result": False,
            },
            [],
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
    minus_self_judge_manifest_jsons: Sequence[Path] = (),
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
                minus_self_judge_manifest_jsons=minus_self_judge_manifest_jsons,
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
    parser.add_argument("--minus-self-judge-manifest", action="append", type=Path, default=[])
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
        minus_self_judge_manifest_jsons=args.minus_self_judge_manifest,
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
