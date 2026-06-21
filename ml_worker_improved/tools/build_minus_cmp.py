#!/usr/bin/env python3
"""Compare rejected middle-minus clips against selected replacement candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.artifact_name_policy import validate_artifact_relative_path, validate_artifact_segment
from tools.docker_runtime import require_docker_runtime
from tools.judge_no_reference_segment_blend_preview import as_float, window_metrics

MANIFEST_NAME = "cmp.json"
ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
IMPROVED_SCORE_DELTA = 0.02
IMPROVED_LEAK_DELTA = -0.02
ACCEPTED_SELECTION_SCHEMAS = {
    "short_listen_repair_selection_pack_manifest.v1",
    "short_candidate_wave_listen_pack_manifest.v1",
    "short_minus_phase_repair_selection_pack_manifest.v1",
}
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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


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
        "expected_acceleration_mode": "external_only_cpu_no_reference_comparison",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "minus_cmp_reads_existing_audio_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "minus_cmp_no_training",
    }
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


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


def rel_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def item_audio_path(item: dict[str, Any], pack_root: Path) -> tuple[str, Path]:
    raw = str(
        item.get("audio_path")
        or item.get("artifact_relative_path")
        or item.get("listen_audio_path")
        or item.get("source_audio_path")
        or ""
    ).strip()
    if not raw:
        raise FileNotFoundError("item audio path missing")
    path = Path(raw)
    if path.is_absolute():
        return raw, path
    validate_artifact_relative_path(raw)
    return raw, pack_root / path


def source_metric_audio_path(item: dict[str, Any], pack_root: Path, reported_path: Path) -> tuple[Path, str]:
    raw = str(item.get("source_audio_path") or "").strip()
    if raw:
        path = Path(raw)
        source_path = path if path.is_absolute() else pack_root / path
        if source_path.exists() and source_path.is_file():
            return source_path, "source_audio_path"
    return reported_path, "reported_audio_path"


def baseline_items(mid_pack: dict[str, Any], baseline_alias: str) -> list[dict[str, Any]]:
    result = []
    for item in mid_pack.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_alias") or "") == baseline_alias:
            result.append(item)
    return sorted(result, key=lambda row: str(row.get("window_id") or ""))


def candidate_items(selection_pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("items", "listen_audio_files"):
        for item in selection_pack.get(key) or []:
            if not isinstance(item, dict):
                continue
            marker = str(
                item.get("artifact_relative_path")
                or item.get("listen_audio_path")
                or item.get("source_audio_path")
                or item.get("source_candidate_id")
                or ""
            )
            if marker and marker in seen:
                continue
            if marker:
                seen.add(marker)
            rows.append(item)
    return rows


def metric_value(metrics: dict[str, Any], key: str) -> float:
    if key in metrics:
        return as_float(metrics.get(key))
    if key == "minus_score" and "overall_no_reference_score" in metrics:
        return as_float(metrics.get("overall_no_reference_score"))
    nested = metrics.get("minus_score_components")
    if isinstance(nested, dict) and key in nested:
        return as_float(nested.get(key))
    return 0.0


def classify(score_delta: float, leak_delta: float, consistency_delta: float) -> str:
    if score_delta >= IMPROVED_SCORE_DELTA and leak_delta <= IMPROVED_LEAK_DELTA:
        return "candidate_proxy_improved"
    if score_delta < -IMPROVED_SCORE_DELTA or leak_delta > abs(IMPROVED_LEAK_DELTA):
        return "candidate_proxy_regressed"
    if consistency_delta < -IMPROVED_SCORE_DELTA:
        return "candidate_proxy_detail_regressed"
    return "candidate_proxy_inconclusive"


def is_regressed_decision(decision: str) -> bool:
    decision = str(decision or "")
    return decision.startswith("candidate_proxy_") and "regressed" in decision


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


def blocked_manifest(
    *,
    output_dir: Path,
    mid_pack_json: Path,
    selection_pack_json: Path,
    failed_checks: Sequence[str],
    command_args: list[str] | None,
) -> dict[str, Any]:
    manifest = {
        "schema": "minus_middle_comparison_report.v1",
        "comparison_status": "blocked",
        "mid_pack_json": str(mid_pack_json),
        "selection_pack_json": str(selection_pack_json),
        "row_count": 0,
        "improved_count": 0,
        "regressed_count": 0,
        "inconclusive_count": 0,
        "failed_checks": stable_unique(list(failed_checks)),
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(output_dir / PROGRESS_NAME),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_comparison_not_for_listen_pack",
        "safety": safety_flags(),
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, [])
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, []), encoding="utf-8")
    return manifest


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# minus middle comparison",
        "",
        f"- comparison_status: `{manifest['comparison_status']}`",
        f"- row_count: `{manifest['row_count']}`",
        f"- improved_count: `{manifest['improved_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['window_id']}` `{row['candidate_audio_path']}` "
            f"`{row['comparison_decision']}` score_delta={row['minus_score_delta']}"
        )
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_minus_cmp(
    *,
    mid_pack_json: Path,
    selection_pack_json: Path,
    output_dir: Path,
    mix_path: Path,
    vocal_path: Path,
    baseline_alias: str = "old",
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", mid_pack_json=str(mid_pack_json), selection_pack_json=str(selection_pack_json))

    failed: list[str] = []
    mid_pack = read_json(mid_pack_json)
    selection_pack = read_json(selection_pack_json)
    failed.extend(collect_true_flag_failures("mid_pack", mid_pack))
    failed.extend(collect_true_flag_failures("selection_pack", selection_pack))
    if mid_pack.get("schema") != "minus_mid_audition_pack.v1":
        failed.append("mid_pack_schema_unexpected")
    if mid_pack.get("pack_status") != "ready_for_diagnostic_listen":
        failed.append("mid_pack_status_not_ready")
    if mid_pack.get("failed_checks") not in ([], None):
        failed.append("mid_pack_failed_checks_not_empty")
    if selection_pack.get("schema") not in ACCEPTED_SELECTION_SCHEMAS:
        failed.append("selection_pack_schema_unexpected")
    if selection_pack.get("delivery_status") != "accepted":
        failed.append("selection_pack_delivery_status_not_accepted")
    if selection_pack.get("delivery_allowed") is not True:
        failed.append("selection_pack_delivery_allowed_not_true")
    if selection_pack.get("failed_checks") not in ([], None):
        failed.append("selection_pack_failed_checks_not_empty")
    if not Path(mix_path).exists():
        failed.append(f"mix_path_missing:{mix_path}")
    if not Path(vocal_path).exists():
        failed.append(f"vocal_path_missing:{vocal_path}")

    baseline = baseline_items(mid_pack, baseline_alias)
    candidates = candidate_items(selection_pack)
    if not baseline:
        failed.append(f"baseline_items_missing:{baseline_alias}")
    if not candidates:
        failed.append("selection_candidates_missing")
    for item in baseline + candidates:
        failed.extend(collect_true_flag_failures("item", item))

    if failed:
        manifest = blocked_manifest(
            output_dir=output_dir,
            mid_pack_json=mid_pack_json,
            selection_pack_json=selection_pack_json,
            failed_checks=failed,
            command_args=command_args,
        )
        progress(progress_path, "finish", comparison_status=manifest["comparison_status"], failed_checks=manifest["failed_checks"])
        return manifest

    rows: list[dict[str, Any]] = []
    render_failed: list[str] = []
    for base_item in baseline:
        window_id = str(base_item.get("window_id") or "")
        start_sec = as_float(base_item.get("window_start_sec"))
        duration_sec = as_float(base_item.get("window_duration_sec"), 12.0)
        try:
            validate_artifact_segment(window_id)
            baseline_rel, baseline_reported_path = item_audio_path(base_item, mid_pack_json.parent)
            baseline_path, baseline_metric_basis = source_metric_audio_path(
                base_item,
                mid_pack_json.parent,
                baseline_reported_path,
            )
            baseline_metric_start_sec = start_sec if baseline_metric_basis == "source_audio_path" else 0.0
            base_metrics = window_metrics(
                mix_path=str(mix_path),
                vocal_path=str(vocal_path),
                minus_path=str(baseline_path),
                start_sec=baseline_metric_start_sec,
                duration_sec=duration_sec,
            )
        except Exception as exc:
            render_failed.append(f"baseline_metric_failed:{window_id}:{type(exc).__name__}:{exc}")
            continue
        for candidate in candidates:
            try:
                candidate_rel, candidate_path = item_audio_path(candidate, selection_pack_json.parent)
                candidate_metrics = window_metrics(
                    mix_path=str(mix_path),
                    vocal_path=str(vocal_path),
                    minus_path=str(candidate_path),
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                )
            except Exception as exc:
                render_failed.append(f"candidate_metric_failed:{window_id}:{type(exc).__name__}:{exc}")
                continue
            base_score = metric_value(base_metrics, "minus_score")
            cand_score = metric_value(candidate_metrics, "minus_score")
            base_leak = metric_value(base_metrics, "vocal_leakage_in_minus_score")
            cand_leak = metric_value(candidate_metrics, "vocal_leakage_in_minus_score")
            base_consistency = metric_value(base_metrics, "mixture_consistency_score")
            cand_consistency = metric_value(candidate_metrics, "mixture_consistency_score")
            score_delta = round(cand_score - base_score, 6)
            leak_delta = round(cand_leak - base_leak, 6)
            consistency_delta = round(cand_consistency - base_consistency, 6)
            rows.append(
                {
                    "schema": "minus_middle_comparison_row.v1",
                    "window_id": window_id,
                    "window_start_sec": round(start_sec, 6),
                    "window_duration_sec": round(duration_sec, 6),
                    "baseline_alias": baseline_alias,
                    "baseline_audio_path": baseline_rel,
                    "baseline_metric_audio_path": str(baseline_path),
                    "baseline_metric_audio_source": baseline_metric_basis,
                    "candidate_audio_path": candidate_rel,
                    "candidate_metric_audio_path": str(candidate_path),
                    "candidate_id": str(candidate.get("source_candidate_id") or candidate.get("candidate_id") or ""),
                    "baseline_minus_score": round(base_score, 6),
                    "candidate_minus_score": round(cand_score, 6),
                    "minus_score_delta": score_delta,
                    "baseline_vocal_leakage": round(base_leak, 6),
                    "candidate_vocal_leakage": round(cand_leak, 6),
                    "vocal_leakage_delta": leak_delta,
                    "baseline_mixture_consistency": round(base_consistency, 6),
                    "candidate_mixture_consistency": round(cand_consistency, 6),
                    "mixture_consistency_delta": consistency_delta,
                    "comparison_decision": classify(score_delta, leak_delta, consistency_delta),
                    "release_ready": False,
                    "can_publish_as_result": False,
                }
            )
            progress(progress_path, "row", window_id=window_id, candidate_audio_path=candidate_rel, comparison_decision=rows[-1]["comparison_decision"])

    if render_failed:
        manifest = blocked_manifest(
            output_dir=output_dir,
            mid_pack_json=mid_pack_json,
            selection_pack_json=selection_pack_json,
            failed_checks=render_failed,
            command_args=command_args,
        )
        progress(progress_path, "finish", comparison_status=manifest["comparison_status"], failed_checks=manifest["failed_checks"])
        return manifest

    improved = sum(1 for row in rows if row["comparison_decision"] == "candidate_proxy_improved")
    regressed = sum(1 for row in rows if is_regressed_decision(row["comparison_decision"]))
    inconclusive = len(rows) - improved - regressed
    manifest = {
        "schema": "minus_middle_comparison_report.v1",
        "comparison_status": "completed",
        "mid_pack_json": str(mid_pack_json),
        "selection_pack_json": str(selection_pack_json),
        "mix_path": str(mix_path),
        "vocal_path": str(vocal_path),
        "row_count": len(rows),
        "improved_count": improved,
        "regressed_count": regressed,
        "inconclusive_count": inconclusive,
        "failed_checks": [],
        "rows_jsonl": str(output_dir / ROWS_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "diagnostic_comparison_not_for_listen_pack",
        "not_claimed": ["release_ready", "publishable_result", "human_listen_verdict", "semantic_clean_minus"],
        "safety": safety_flags(),
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / ROWS_NAME, rows)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, rows), encoding="utf-8")
    progress(progress_path, "finish", comparison_status=manifest["comparison_status"], row_count=len(rows), failed_checks=[])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rejected middle-minus clips with replacement candidates.")
    parser.add_argument("--mid-pack-json", type=Path, required=True)
    parser.add_argument("--selection-pack-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix-path", type=Path, required=True)
    parser.add_argument("--vocal-path", type=Path, required=True)
    parser.add_argument("--baseline-alias", default="old")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(tool_name="build_minus_cmp.py", entrypoint_path="/app/tools/build_minus_cmp.py")
    args = parse_args(argv)
    manifest = build_minus_cmp(
        mid_pack_json=args.mid_pack_json,
        selection_pack_json=args.selection_pack_json,
        output_dir=args.output_dir,
        mix_path=args.mix_path,
        vocal_path=args.vocal_path,
        baseline_alias=str(args.baseline_alias),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "comparison_status": manifest["comparison_status"],
                "row_count": manifest["row_count"],
                "improved_count": manifest["improved_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["comparison_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
