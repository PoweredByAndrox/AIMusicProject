#!/usr/bin/env python3
"""Build a short-name listen pack from accepted candidate-wave minus rows."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifact_name_policy import ARTIFACT_AUDIO_DIR, indexed_audio_filename, validate_artifact_segment  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402


MANIFEST_NAME = "pack.json"
SELECTED_NAME = "selected.jsonl"
BLOCKED_NAME = "blocked.jsonl"
SUMMARY_NAME = "summary.md"
PROGRESS_NAME = "progress.jsonl"
START_TIME = time.monotonic()
DIAGNOSTIC_DELIVERY_STATUS = "blocked"
DIAGNOSTIC_PRESENTATION_STATUS = "diagnostic_only_not_for_listen_pack"
DIAGNOSTIC_RELEASE_STATUS = "not_release_ready"
DIAGNOSTIC_BLOCKERS = [
    "candidate_wave_pack_is_diagnostic_only",
    "requires_staged_aggregate_machine_gate_for_listen_delivery",
]

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def progress(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event": event,
        "elapsed_sec": round(time.monotonic() - START_TIME, 6),
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "candidate_wave_listen_pack_reads_existing_audio_only",
        "acceleration_mode": "external_only_cpu_dsp_packaging",
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


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


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("source_candidate_id") or "")


def minus_path(row: dict[str, Any]) -> Path:
    return Path(str(row.get("minus_path") or row.get("instrumental_path") or ""))


def load_judge_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id") or ""): row for row in read_jsonl(path) if row.get("candidate_id")}


def judge_accepts(candidate_id_value: str, result_rows: dict[str, dict[str, Any]], windowed_rows: dict[str, dict[str, Any]]) -> bool:
    result = result_rows.get(candidate_id_value, {})
    windowed = windowed_rows.get(candidate_id_value, {})
    return (
        result.get("decision") == "accept_candidate"
        and result.get("minus_decision") == "accept_candidate"
        and windowed.get("decision") == "accept_candidate"
        and windowed.get("minus_decision") == "accept_candidate"
    )


def judge_reasons(candidate_id_value: str, result_rows: dict[str, dict[str, Any]], windowed_rows: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for row in (result_rows.get(candidate_id_value, {}), windowed_rows.get(candidate_id_value, {})):
        reasons.extend(str(reason) for reason in row.get("reasons") or [] if str(reason))
        decision = str(row.get("decision") or "")
        minus_decision = str(row.get("minus_decision") or "")
        if decision and decision != "accept_candidate":
            reasons.append(decision)
        if minus_decision and minus_decision != "accept_candidate":
            reasons.append(minus_decision)
    return stable_unique(reasons)


def sort_key(row: dict[str, Any], result_rows: dict[str, dict[str, Any]], windowed_rows: dict[str, dict[str, Any]]) -> tuple[float, float, float, float, str]:
    cid = candidate_id(row)
    result = result_rows.get(cid, {})
    windowed = windowed_rows.get(cid, {})
    windowed_score = float(windowed.get("minus_score") or 0.0)
    result_score = float(result.get("minus_score") or 0.0)
    leakage = float(windowed.get("vocal_leakage_in_minus_score") or result.get("vocal_leakage_in_minus_score") or 1.0)
    mixture = float(windowed.get("mixture_consistency_score") or 0.0)
    return (windowed_score, result_score, -leakage, mixture, cid)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def warning_comparison_gate(path: Path) -> tuple[dict[str, Any], list[str]]:
    comparison = read_json(path)
    failed: list[str] = []
    failed.extend(collect_true_flag_failures("minus_warning_comparison", comparison))
    if comparison.get("schema") != "minus_middle_comparison_report.v1":
        failed.append("minus_warning_comparison_schema_unexpected")
    if comparison.get("comparison_status") != "completed":
        failed.append("minus_warning_comparison_not_completed")
    if comparison.get("failed_checks") not in ([], None):
        failed.append("minus_warning_comparison_failed_checks")
    improved_count = as_int(comparison.get("improved_count"))
    regressed_count = as_int(comparison.get("regressed_count"))
    if improved_count < 1:
        failed.append("minus_warning_comparison_no_improved_windows")
    if regressed_count > 0:
        failed.append("minus_warning_comparison_regressed_windows")
    return (
        {
            "schema": str(comparison.get("schema") or ""),
            "comparison_status": str(comparison.get("comparison_status") or ""),
            "improved_count": improved_count,
            "regressed_count": regressed_count,
            "failed_checks": list(comparison.get("failed_checks") or []),
            "release_ready": bool(comparison.get("release_ready") is True),
            "can_publish_as_result": bool(comparison.get("can_publish_as_result") is True),
        },
        stable_unique(failed),
    )


def copy_minus_audio(
    *,
    row: dict[str, Any],
    output_dir: Path,
    index: int,
    result_judge: dict[str, Any],
    windowed_judge: dict[str, Any],
) -> dict[str, Any]:
    validate_artifact_segment("min")
    source_path = minus_path(row)
    filename = indexed_audio_filename("m", index)
    target_path = output_dir / ARTIFACT_AUDIO_DIR / "min" / filename
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(candidate_id(row))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    metadata = audio_metadata(target_path)
    return {
        "schema": "short_candidate_wave_listen_audio.v1",
        "selection_rank": index,
        "route_intent": "candidate_wave_minus_ab",
        "stem_type": "minus",
        "source_alias": "cw",
        "track_alias": "t001",
        "artifact_alias": filename.removesuffix(".flac"),
        "artifact_filename": filename,
        "artifact_relative_path": str(target_path.relative_to(output_dir).as_posix()),
        "listen_audio_path": str(target_path),
        "listen_audio_sha256": sha256_file(target_path),
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "duration_sec": metadata["duration_sec"],
        "source_audio_path": str(source_path),
        "source_candidate_id": candidate_id(row),
        "source_track_id": str(row.get("track_id") or ""),
        "source_route_id": str(row.get("route_id") or ""),
        "source_candidate_family": str(row.get("candidate_family") or ""),
        "source_row": row,
        "minus_judge": {
            "result": result_judge,
            "windowed": windowed_judge,
        },
        "quality_warning_reasons": [str(reason) for reason in windowed_judge.get("reasons") or [] if str(reason)],
        "selection_status": "selected_for_diagnostic_ab",
        "delivery_status": DIAGNOSTIC_DELIVERY_STATUS,
        "delivery_allowed": False,
        "presentation_status": DIAGNOSTIC_PRESENTATION_STATUS,
        "release_status": DIAGNOSTIC_RELEASE_STATUS,
        "delivery_blockers": list(DIAGNOSTIC_BLOCKERS),
        "release_ready": False,
        "can_publish_as_result": False,
        "listenable": False,
        "semantic_quality_verified": False,
    }


def summary_text(manifest: dict[str, Any], selected: Sequence[dict[str, Any]], blocked: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short candidate wave listen pack",
        "",
        f"- delivery_status: `{manifest['delivery_status']}`",
        f"- selected_count: `{manifest['selected_count']}`",
        f"- blocked_candidate_count: `{manifest['blocked_candidate_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        "",
        "## Selected",
        "",
    ]
    for row in selected:
        lines.append(f"- `{row['artifact_relative_path']}` rank `{row['selection_rank']}`")
    if blocked:
        lines.extend(["", "## Blocked / Not Selected", ""])
        for index, row in enumerate(blocked, start=1):
            lines.append(f"- blocked `{index:03d}` `{','.join(row.get('block_reasons') or [])}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_short_candidate_wave_listen_pack(
    *,
    candidate_pool_jsonl: Path,
    minus_result_judge_jsonl: Path,
    minus_windowed_judge_jsonl: Path,
    output_dir: Path,
    max_minus_candidates: int = 4,
    minus_warning_comparison_json: Path | None = None,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", candidate_pool_jsonl=str(candidate_pool_jsonl))

    failed: list[str] = []
    for required_path, label in (
        (candidate_pool_jsonl, "candidate_pool_jsonl"),
        (minus_result_judge_jsonl, "minus_result_judge_jsonl"),
        (minus_windowed_judge_jsonl, "minus_windowed_judge_jsonl"),
    ):
        if not required_path.exists():
            failed.append(f"{label}_missing")
    if minus_warning_comparison_json is not None and not minus_warning_comparison_json.exists():
        failed.append("minus_warning_comparison_json_missing")

    selected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    warning_comparison: dict[str, Any] = {}
    if not failed:
        pool_rows = read_jsonl(candidate_pool_jsonl)
        result_rows = load_judge_rows(minus_result_judge_jsonl)
        windowed_rows = load_judge_rows(minus_windowed_judge_jsonl)
        if minus_warning_comparison_json is not None:
            warning_comparison, comparison_failures = warning_comparison_gate(minus_warning_comparison_json)
            failed.extend(comparison_failures)
            progress(
                progress_path,
                "warning_comparison_gate",
                minus_warning_comparison_json=str(minus_warning_comparison_json),
                failed_checks=comparison_failures,
                improved_count=warning_comparison.get("improved_count"),
                regressed_count=warning_comparison.get("regressed_count"),
            )
        for row in pool_rows:
            cid = candidate_id(row)
            for check in collect_true_flag_failures("candidate_pool_row", row):
                failed.append(f"{check}:{cid}")
            if row.get("is_complete_pair") is False:
                blocked.append({"source_candidate_id": cid, "block_reasons": ["incomplete_pair"], "source_row": row})
                continue
            if not judge_accepts(cid, result_rows, windowed_rows):
                blocked.append({"source_candidate_id": cid, "block_reasons": judge_reasons(cid, result_rows, windowed_rows) or ["minus_judge_not_accepted"], "source_row": row})
                continue
        if not failed:
            accepted_rows = [
                row
                for row in pool_rows
                if row.get("is_complete_pair") is not False and judge_accepts(candidate_id(row), result_rows, windowed_rows)
            ]
            accepted_rows.sort(key=lambda row: sort_key(row, result_rows, windowed_rows), reverse=True)
            if max_minus_candidates > 0:
                accepted_rows = accepted_rows[:max_minus_candidates]
            if not accepted_rows:
                failed.append("no_accepted_candidate_wave_minus_rows")
            for index, row in enumerate(accepted_rows, start=1):
                cid = candidate_id(row)
                try:
                    selected.append(
                        copy_minus_audio(
                            row=row,
                            output_dir=output_dir,
                            index=index,
                            result_judge=result_rows.get(cid, {}),
                            windowed_judge=windowed_rows.get(cid, {}),
                        )
                    )
                except FileNotFoundError:
                    failed.append(f"selected_audio_missing:{cid}")
                    break
                progress(progress_path, "copy_selected", index=index, candidate_id=cid)

    failed = stable_unique(failed)
    selection_ready = bool(selected) and not failed
    if not selection_ready:
        selected = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    manifest = {
        "schema": "short_candidate_wave_listen_pack_manifest.v1",
        "pack_id": output_dir.name or "cw_listen",
        "selection_status": "selected_for_diagnostic_ab" if selection_ready else "blocked",
        "delivery_status": DIAGNOSTIC_DELIVERY_STATUS,
        "delivery_allowed": False,
        "presentation_status": DIAGNOSTIC_PRESENTATION_STATUS if selection_ready else "blocked_from_listen_pack",
        "release_status": DIAGNOSTIC_RELEASE_STATUS,
        "delivery_blockers": list(DIAGNOSTIC_BLOCKERS) if selection_ready else [],
        "release_ready": False,
        "can_publish_as_result": False,
        "deliverable": False,
        "listenable": False,
        "selected_count": len(selected),
        "blocked_candidate_count": len(blocked),
        "listen_audio_files": selected,
        "candidate_pool_jsonl": str(candidate_pool_jsonl),
        "minus_result_judge_jsonl": str(minus_result_judge_jsonl),
        "minus_windowed_judge_jsonl": str(minus_windowed_judge_jsonl),
        "minus_warning_comparison_json": str(minus_warning_comparison_json or ""),
        "minus_warning_comparison": warning_comparison,
        "failed_checks": failed,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "blocked_jsonl": str(output_dir / BLOCKED_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "group_dir_strategy": "min",
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
            "blocked_candidates_excluded": True,
            "dirty_stems_never_ground_truth": True,
        },
        "safety": {
            "reference_required": False,
            "human_csv_required": False,
            "training_started": False,
            "separator_training_started": False,
            "repair_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "thresholds_relaxed": False,
            "cleanup_routes_added": False,
            "release_ready": False,
        },
        "not_claimed": ["release_ready", "publishable_result", "audio_repaired", "semantic_ground_truth"],
        "command_args": command_args or [],
    }
    write_jsonl(output_dir / SELECTED_NAME, selected)
    write_jsonl(output_dir / BLOCKED_NAME, blocked)
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, selected, blocked), encoding="utf-8")
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], selected_count=len(selected), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short listen pack from accepted candidate-wave minus rows.")
    parser.add_argument("--candidate-pool-jsonl", type=Path, required=True)
    parser.add_argument("--minus-result-judge-jsonl", type=Path, required=True)
    parser.add_argument("--minus-windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-minus-candidates", type=int, default=4)
    parser.add_argument("--minus-warning-comparison-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_candidate_wave_listen_pack.py",
        entrypoint_path="/app/tools/build_short_candidate_wave_listen_pack.py",
    )
    args = parse_args(argv)
    manifest = build_short_candidate_wave_listen_pack(
        candidate_pool_jsonl=args.candidate_pool_jsonl,
        minus_result_judge_jsonl=args.minus_result_judge_jsonl,
        minus_windowed_judge_jsonl=args.minus_windowed_judge_jsonl,
        output_dir=args.output_dir,
        max_minus_candidates=int(args.max_minus_candidates),
        minus_warning_comparison_json=args.minus_warning_comparison_json,
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "selected_count": manifest["selected_count"],
                "blocked_candidate_count": manifest["blocked_candidate_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not manifest["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
