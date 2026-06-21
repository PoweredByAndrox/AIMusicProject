#!/usr/bin/env python3
"""Build full-track segment-switch inputs from a minus comparison report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_relative_path  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402

MANIFEST_NAME = "manifest.json"
POOL_NAME = "pool.jsonl"
WINDOWED_JUDGE_NAME = "wj.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
BLOCKED_TRUE_FLAGS = (
    "release_ready",
    "can_publish_as_result",
    "training_started",
    "separator_training_started",
    "models_changed",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
)
ACCEPTED_SELECTION_SCHEMAS = {
    "short_listen_repair_selection_pack_manifest.v1",
    "short_candidate_wave_listen_pack_manifest.v1",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def progress(path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 3),
        "expected_acceleration_mode": "external_only_cpu_manifest_build",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "cmp_switch_inputs_reads_existing_manifests_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "cmp_switch_inputs_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
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
    contract = payload.get("contract")
    if isinstance(contract, dict):
        for key in BLOCKED_TRUE_FLAGS:
            if contract.get(key) is True:
                failed.append(f"{prefix}_contract_{key}_true")
    return failed


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rows_path_from(cmp_manifest: dict[str, Any], cmp_json: Path) -> Path:
    raw = str(cmp_manifest.get("rows_jsonl") or "").strip()
    if raw:
        return Path(raw)
    return cmp_json.parent / "rows.jsonl"


def selection_rows(selection_pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("listen_audio_files", "items"):
        for row in selection_pack.get(key) or []:
            if not isinstance(row, dict):
                continue
            marker = str(
                row.get("artifact_relative_path")
                or row.get("listen_audio_path")
                or row.get("source_audio_path")
                or row.get("source_candidate_id")
                or ""
            )
            if marker and marker in seen:
                continue
            if marker:
                seen.add(marker)
            rows.append(row)
    return rows


def selection_maps(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        artifact_path = str(row.get("artifact_relative_path") or "")
        if artifact_path:
            by_path[artifact_path] = row
        candidate_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "")
        if candidate_id:
            by_candidate[candidate_id] = row
    return by_path, by_candidate


def selected_audio_path(row: dict[str, Any], selection_pack_json: Path) -> str:
    raw = str(row.get("listen_audio_path") or row.get("source_audio_path") or row.get("artifact_relative_path") or "")
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute():
        return raw
    validate_artifact_relative_path(raw)
    return str(selection_pack_json.parent / path)


def window_key(row: dict[str, Any]) -> tuple[float, float]:
    return (round(as_float(row.get("window_start_sec")), 6), round(as_float(row.get("window_duration_sec")), 6))


def base_window_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": str(row.get("window_id") or ""),
        "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
        "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
        "minus_score": round(as_float(row.get("baseline_minus_score")), 6),
        "window_fitness": round(as_float(row.get("baseline_minus_score")), 6),
        "vocal_leakage_in_minus_score": round(as_float(row.get("baseline_vocal_leakage")), 6),
        "mixture_consistency_score": round(as_float(row.get("baseline_mixture_consistency")), 6),
        "cmp_selection_decision": "accept_candidate",
        "comparison_decision": "baseline_kept",
        "reasons": ["cmp_baseline_available"],
    }


def candidate_window_metric(row: dict[str, Any]) -> dict[str, Any]:
    decision = str(row.get("comparison_decision") or "")
    accepted = decision == "candidate_proxy_improved"
    reasons = ["no_reference_proxy_passed"] if accepted else [decision or "candidate_proxy_not_improved"]
    return {
        "window_id": str(row.get("window_id") or ""),
        "window_start_sec": round(as_float(row.get("window_start_sec")), 6),
        "window_duration_sec": round(as_float(row.get("window_duration_sec")), 6),
        "minus_score": round(as_float(row.get("candidate_minus_score")), 6),
        "window_fitness": round(as_float(row.get("candidate_minus_score")), 6),
        "vocal_leakage_in_minus_score": round(as_float(row.get("candidate_vocal_leakage")), 6),
        "mixture_consistency_score": round(as_float(row.get("candidate_mixture_consistency")), 6),
        "cmp_selection_decision": "accept_candidate" if accepted else "blocked_hard_case",
        "comparison_decision": decision,
        "reasons": reasons,
    }


def safety_flags() -> dict[str, bool]:
    return {
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "models_changed": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "release_ready": False,
    }


def summary_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# minus cmp switch inputs",
        "",
        f"- input_status: `{manifest['input_status']}`",
        f"- pool_row_count: `{manifest['pool_row_count']}`",
        f"- windowed_judge_row_count: `{manifest['windowed_judge_row_count']}`",
        f"- improved_window_count: `{manifest['improved_window_count']}`",
        f"- blocked_alternate_window_count: `{manifest['blocked_alternate_window_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
    ]
    if manifest["failed_checks"]:
        lines.extend(["## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    return "\n".join(lines) + "\n"


def build_minus_cmp_switch_inputs(
    *,
    cmp_json: Path,
    selection_pack_json: Path,
    output_dir: Path,
    track_id: str,
    mix_path: Path,
    vocal_path: Path,
    base_candidate_id: str = "old",
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", cmp_json=str(cmp_json), selection_pack_json=str(selection_pack_json))

    cmp_manifest = read_json(cmp_json)
    selection_pack = read_json(selection_pack_json)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("cmp", cmp_manifest))
    failed.extend(collect_true_flag_failures("selection_pack", selection_pack))
    if cmp_manifest.get("schema") != "minus_middle_comparison_report.v1":
        failed.append("cmp_schema_unexpected")
    if cmp_manifest.get("comparison_status") != "completed":
        failed.append("cmp_status_not_completed")
    if cmp_manifest.get("failed_checks") not in ([], None):
        failed.append("cmp_failed_checks_not_empty")
    if selection_pack.get("schema") not in ACCEPTED_SELECTION_SCHEMAS:
        failed.append("selection_pack_schema_unexpected")
    if selection_pack.get("delivery_status") != "accepted" or selection_pack.get("delivery_allowed") is not True:
        failed.append("selection_pack_not_accepted")
    if selection_pack.get("failed_checks") not in ([], None):
        failed.append("selection_pack_failed_checks_not_empty")

    cmp_rows_path = rows_path_from(cmp_manifest, cmp_json)
    if not cmp_rows_path.exists():
        failed.append(f"cmp_rows_jsonl_missing:{cmp_rows_path}")
        cmp_rows: list[dict[str, Any]] = []
    else:
        cmp_rows = iter_jsonl(cmp_rows_path)
    if not cmp_rows:
        failed.append("cmp_rows_empty")

    rows = selection_rows(selection_pack)
    by_path, by_candidate = selection_maps(rows)
    baseline_minus_path = ""
    pool_rows: list[dict[str, Any]] = []
    windowed_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    improved_windows = 0
    blocked_alternate_windows = 0

    if not failed:
        first_row = cmp_rows[0]
        baseline_minus_path = str(first_row.get("baseline_metric_audio_path") or "").strip()
        if not baseline_minus_path:
            failed.append("baseline_metric_audio_path_missing")
        else:
            pool_rows.append(
                {
                    "track_id": track_id,
                    "candidate_id": base_candidate_id,
                    "mix_path": str(mix_path),
                    "vocal_path": str(vocal_path),
                    "minus_path": baseline_minus_path,
                    "instrumental_path": baseline_minus_path,
                    "source_minus_path": baseline_minus_path,
                    "base_minus_path": baseline_minus_path,
                    "candidate_family": "cmp_baseline_minus",
                    "candidate_origin": "minus_cmp_baseline",
                    "delivery_status": "blocked",
                    "delivery_allowed": False,
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            )
        seen_base_windows: set[tuple[float, float]] = set()
        candidate_ids_seen: set[str] = set()
        for row in cmp_rows:
            key = window_key(row)
            if key not in seen_base_windows:
                windowed_by_candidate[base_candidate_id].append(base_window_metric(row))
                seen_base_windows.add(key)
            candidate_id = str(row.get("candidate_id") or "").strip()
            candidate_audio_path = str(row.get("candidate_audio_path") or "")
            source_row = by_candidate.get(candidate_id) or by_path.get(candidate_audio_path)
            if not candidate_id or source_row is None:
                failed.append(f"candidate_source_missing:{candidate_id or candidate_audio_path}")
                continue
            if candidate_id not in candidate_ids_seen:
                minus_path = selected_audio_path(source_row, selection_pack_json)
                if not minus_path:
                    failed.append(f"candidate_audio_path_missing:{candidate_id}")
                    continue
                candidate_source_minus_path = str(
                    source_row.get("source_minus_path")
                    or source_row.get("base_minus_path")
                    or baseline_minus_path
                )
                pool_rows.append(
                    {
                        "track_id": track_id,
                        "candidate_id": candidate_id,
                        "mix_path": str(mix_path),
                        "vocal_path": str(vocal_path),
                        "minus_path": minus_path,
                        "instrumental_path": minus_path,
                        "source_minus_path": candidate_source_minus_path,
                        "base_minus_path": candidate_source_minus_path,
                        "candidate_family": "cmp_selected_minus_repair",
                        "candidate_origin": "minus_cmp_improved_window_candidate",
                        "source_artifact_relative_path": str(source_row.get("artifact_relative_path") or ""),
                        "source_selection_pack_json": str(selection_pack_json),
                        "delivery_status": "blocked",
                        "delivery_allowed": False,
                        "release_ready": False,
                        "can_publish_as_result": False,
                    }
                )
                candidate_ids_seen.add(candidate_id)
            metric = candidate_window_metric(row)
            if metric["cmp_selection_decision"] == "accept_candidate":
                improved_windows += 1
            else:
                blocked_alternate_windows += 1
            windowed_by_candidate[candidate_id].append(metric)

    failed = stable_unique(failed)
    if failed:
        pool_rows = []
        judge_rows: list[dict[str, Any]] = []
        status = "blocked"
    else:
        judge_rows = [
            {
                "candidate_id": candidate_id,
                "decision": "accept_candidate",
                "minus_decision": "accept_candidate",
                "window_metrics": sorted(metrics, key=lambda item: (item["window_start_sec"], item["window_duration_sec"])),
                "release_ready": False,
                "can_publish_as_result": False,
            }
            for candidate_id, metrics in sorted(windowed_by_candidate.items())
        ]
        status = "ready"

    manifest = {
        "schema": "minus_cmp_switch_inputs.v1",
        "input_status": status,
        "cmp_json": str(cmp_json),
        "cmp_rows_jsonl": str(cmp_rows_path),
        "selection_pack_json": str(selection_pack_json),
        "track_id": track_id,
        "base_candidate_id": base_candidate_id,
        "baseline_minus_path": baseline_minus_path,
        "pool_jsonl": str(output_dir / POOL_NAME),
        "windowed_judge_jsonl": str(output_dir / WINDOWED_JUDGE_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "pool_row_count": len(pool_rows),
        "windowed_judge_row_count": len(judge_rows),
        "alternate_candidate_count": max(0, len(pool_rows) - 1),
        "improved_window_count": improved_windows if not failed else 0,
        "blocked_alternate_window_count": blocked_alternate_windows if not failed else 0,
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_switch_inputs_not_for_listen_pack",
        "renderer_contract": {
            "selection_score_field": "minus_score",
            "selection_decision_field": "cmp_selection_decision",
            "base_candidate_id": base_candidate_id,
            "requires_post_render_judge": True,
            "full_track_renderer_expected": True,
        },
        "safety": safety_flags(),
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / POOL_NAME, pool_rows)
    write_jsonl(output_dir / WINDOWED_JUDGE_NAME, judge_rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest), encoding="utf-8")
    progress(progress_path, "finish", input_status=status, failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build segment-switch inputs from a minus comparison report.")
    parser.add_argument("--cmp-json", type=Path, required=True)
    parser.add_argument("--selection-pack-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--mix-path", type=Path, required=True)
    parser.add_argument("--vocal-path", type=Path, required=True)
    parser.add_argument("--base-candidate-id", default="old")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_minus_cmp_switch_inputs.py", entrypoint_path="/app/tools/build_minus_cmp_switch_inputs.py")
    args = parse_args(argv)
    manifest = build_minus_cmp_switch_inputs(
        cmp_json=args.cmp_json,
        selection_pack_json=args.selection_pack_json,
        output_dir=args.output_dir,
        track_id=str(args.track_id),
        mix_path=args.mix_path,
        vocal_path=args.vocal_path,
        base_candidate_id=str(args.base_candidate_id),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "input_status": manifest["input_status"],
                "pool_row_count": manifest["pool_row_count"],
                "windowed_judge_row_count": manifest["windowed_judge_row_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["input_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
