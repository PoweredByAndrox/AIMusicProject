#!/usr/bin/env python3
"""Build a short listen pack from accepted minus phase-repair variants."""

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
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
        "expected_acceleration_mode": "external_only_cpu_phase_repair_selection_pack",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "phase_repair_selection_pack_copies_existing_judged_variants_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "phase_repair_selection_pack_no_training",
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


def load_rows_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id") or ""): row for row in iter_jsonl(path) if row.get("candidate_id")}


def phase_repair_has_blocking_warning(row: dict[str, Any]) -> bool:
    pair_decision = str(row.get("pair_decision") or "")
    if pair_decision and pair_decision != "pair_compatible":
        return True
    reasons = {str(reason) for reason in row.get("reasons") or [] if str(reason)}
    return bool(reasons.difference({"no_reference_proxy_passed"}))


def accepted_by_judges(candidate_id: str, result_rows: dict[str, dict[str, Any]], windowed_rows: dict[str, dict[str, Any]]) -> bool:
    result = result_rows.get(candidate_id, {})
    windowed = windowed_rows.get(candidate_id, {})
    return (
        result.get("decision") == "accept_candidate"
        and result.get("minus_decision") in (None, "accept_candidate")
        and windowed.get("decision") == "accept_candidate"
        and windowed.get("minus_decision") == "accept_candidate"
        and not phase_repair_has_blocking_warning(result)
        and not phase_repair_has_blocking_warning(windowed)
    )


def sort_key(row: dict[str, Any], windowed_rows: dict[str, dict[str, Any]]) -> tuple[float, float, float, float, str]:
    metrics = windowed_rows.get(str(row.get("candidate_id") or ""), {})
    return (
        float(metrics.get("minus_score") or 0.0),
        float(metrics.get("instrumental_phase_consistency_score") or 0.0),
        float(metrics.get("mixture_consistency_score") or 0.0),
        -float(metrics.get("vocal_leakage_in_minus_score") or 1.0),
        str(row.get("candidate_id") or ""),
    )


def source_audio_path(row: dict[str, Any], windowed: dict[str, Any]) -> Path:
    raw = str(windowed.get("minus_path") or row.get("minus_path") or row.get("instrumental_path") or "")
    if not raw:
        raise FileNotFoundError("minus_path_missing")
    path = Path(raw)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def copy_selected_audio(
    *,
    source_row: dict[str, Any],
    windowed_row: dict[str, Any],
    phase_manifest: dict[str, Any],
    output_dir: Path,
    index: int,
) -> dict[str, Any]:
    validate_artifact_segment("min")
    source_path = source_audio_path(source_row, windowed_row)
    filename = indexed_audio_filename("m", index)
    target_path = output_dir / ARTIFACT_AUDIO_DIR / "min" / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    metadata = audio_metadata(target_path)
    reasons = [str(reason) for reason in windowed_row.get("reasons") or [] if str(reason) and str(reason) != "no_reference_proxy_passed"]
    return {
        "schema": "short_minus_phase_repair_selection_audio.v1",
        "selection_rank": index,
        "route_intent": "minus_phase_repair_selection",
        "stem_type": "minus",
        "artifact_alias": filename.removesuffix(".flac"),
        "artifact_filename": filename,
        "artifact_relative_path": str(target_path.relative_to(output_dir).as_posix()),
        "listen_audio_path": str(target_path),
        "listen_audio_sha256": sha256_file(target_path),
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "duration_sec": metadata["duration_sec"],
        "source_audio_path": str(source_path),
        "source_candidate_id": str(source_row.get("candidate_id") or windowed_row.get("candidate_id") or ""),
        "source_phase_repair_manifest_json": str(phase_manifest.get("manifest_path") or ""),
        "source_phase_repair_manifest": {
            "track_id": str(phase_manifest.get("track_id") or ""),
            "source_candidate_id": str(phase_manifest.get("source_candidate_id") or ""),
            "source_minus_route_id": str(phase_manifest.get("source_minus_route_id") or ""),
            "source_mix_path": str(phase_manifest.get("source_mix_path") or ""),
            "source_vocal_path": str(phase_manifest.get("source_vocal_path") or ""),
            "source_minus_path": str(phase_manifest.get("source_minus_path") or ""),
        },
        "source_row": source_row,
        "selection_basis": "accepted_result_and_windowed_minus_phase_repair_rank",
        "phase_repair_metrics": {
            "minus_score": windowed_row.get("minus_score"),
            "mixture_consistency_score": windowed_row.get("mixture_consistency_score"),
            "vocal_leakage_in_minus_score": windowed_row.get("vocal_leakage_in_minus_score"),
            "instrumental_phase_consistency_score": windowed_row.get("instrumental_phase_consistency_score"),
            "phase_consistency_score": windowed_row.get("phase_consistency_score"),
            "pair_decision": windowed_row.get("pair_decision"),
            "pair_phase_consistency_score": windowed_row.get("pair_phase_consistency_score"),
        },
        "quality_warning_reasons": reasons,
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "semantic_quality_verified": False,
    }


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short minus phase repair selection pack",
        "",
        f"- delivery_status: `{manifest['delivery_status']}`",
        f"- selected_count: `{manifest['selected_count']}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        f"- can_publish_as_result: `{str(manifest['can_publish_as_result']).lower()}`",
        f"- quality_warning_count: `{manifest['quality_warning_count']}`",
        "",
        "## Selected Audio",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['artifact_relative_path']}` `{row['source_candidate_id']}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_short_minus_phase_repair_selection_pack(
    *,
    phase_repair_manifest_json: Path,
    result_judge_jsonl: Path,
    windowed_judge_jsonl: Path,
    output_dir: Path,
    max_candidates: int = 1,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", phase_repair_manifest_json=str(phase_repair_manifest_json))

    phase_manifest = read_json(phase_repair_manifest_json)
    phase_manifest["manifest_path"] = str(phase_repair_manifest_json)
    failed = collect_true_flag_failures("phase_repair", phase_manifest)
    if phase_manifest.get("schema") != "no_reference_minus_phase_repair_manifest.v1":
        failed.append("phase_repair_schema_unexpected")
    if phase_manifest.get("delivery_status") != "diagnostic_only":
        failed.append("phase_repair_delivery_status_not_diagnostic_only")
    if phase_manifest.get("delivery_allowed") is not False:
        failed.append("phase_repair_delivery_allowed_not_false")
    candidate_pair_jsonl = Path(str(phase_manifest.get("candidate_pair_jsonl") or ""))
    if not candidate_pair_jsonl.exists():
        failed.append("phase_repair_candidate_pair_missing")
    if not result_judge_jsonl.exists():
        failed.append("result_judge_missing")
    if not windowed_judge_jsonl.exists():
        failed.append("windowed_judge_missing")

    selected: list[dict[str, Any]] = []
    if not failed:
        candidate_rows = list(iter_jsonl(candidate_pair_jsonl))
        result_rows = load_rows_by_candidate(result_judge_jsonl)
        windowed_rows = load_rows_by_candidate(windowed_judge_jsonl)
        accepted_rows = [
            row
            for row in candidate_rows
            if accepted_by_judges(str(row.get("candidate_id") or ""), result_rows, windowed_rows)
            and row.get("release_ready") is not True
            and row.get("can_publish_as_result") is not True
            and row.get("can_publish_as_release") is not True
            and row.get("can_mark_release_ready") is not True
        ]
        accepted_rows = sorted(accepted_rows, key=lambda row: sort_key(row, windowed_rows), reverse=True)[: max(0, max_candidates)]
        if not accepted_rows:
            failed.append("no_accepted_phase_repair_variants")
        for index, row in enumerate(accepted_rows, start=1):
            selected.append(
                copy_selected_audio(
                    source_row=row,
                    windowed_row=windowed_rows[str(row.get("candidate_id") or "")],
                    phase_manifest=phase_manifest,
                    output_dir=output_dir,
                    index=index,
                )
            )

    failed = stable_unique(failed)
    accepted = bool(selected) and not failed
    if not accepted:
        selected = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    manifest = {
        "schema": "short_minus_phase_repair_selection_pack_manifest.v1",
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
        "quality_warning_count": sum(1 for row in selected if row.get("quality_warning_reasons")),
        "source_phase_repair_manifest_json": str(phase_repair_manifest_json),
        "result_judge_jsonl": str(result_judge_jsonl),
        "windowed_judge_jsonl": str(windowed_judge_jsonl),
        "failed_checks": failed,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
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
    write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUMMARY_NAME).write_text(summary_text(manifest, selected), encoding="utf-8")
    progress(progress_path, "finish", delivery_status=manifest["delivery_status"], selected_count=len(selected), failed_checks=failed)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short listen pack from accepted minus phase-repair variants.")
    parser.add_argument("--phase-repair-manifest-json", type=Path, required=True)
    parser.add_argument("--result-judge-jsonl", type=Path, required=True)
    parser.add_argument("--windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_minus_phase_repair_selection_pack.py",
        entrypoint_path="/app/tools/build_short_minus_phase_repair_selection_pack.py",
    )
    args = parse_args(argv)
    manifest = build_short_minus_phase_repair_selection_pack(
        phase_repair_manifest_json=args.phase_repair_manifest_json,
        result_judge_jsonl=args.result_judge_jsonl,
        windowed_judge_jsonl=args.windowed_judge_jsonl,
        output_dir=args.output_dir,
        max_candidates=int(args.max_candidates),
        command_args=list(argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.output_dir / MANIFEST_NAME),
                "delivery_status": manifest["delivery_status"],
                "selected_count": manifest["selected_count"],
                "failed_checks": manifest["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["delivery_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
