#!/usr/bin/env python3
"""Select a concise short-name repair listen pack from repair candidates."""

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
        "expected_acceleration_mode": "external_only_cpu_repair_selection_pack",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "repair_selection_pack_copies_existing_candidates_only",
        "gpu_training_skipped": True,
        "gpu_training_skip_reason": "repair_selection_pack_no_training",
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


def accepted_candidate_row(row: dict[str, Any]) -> bool:
    return (
        row.get("delivery_status") == "accepted"
        and row.get("delivery_allowed") is True
        and row.get("release_ready") is not True
        and row.get("can_publish_as_result") is not True
    )


def reference_sdr(row: dict[str, Any]) -> float:
    metrics = row.get("reference_metrics") if isinstance(row.get("reference_metrics"), dict) else {}
    value = metrics.get("reference_sdr_db")
    return float(value) if isinstance(value, (int, float)) else -1.0e12


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("source_candidate_id") or "")


def source_dicts(row: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [row]
    for key in ("source_row", "source_phase_repair_manifest", "source_manifest"):
        value = row.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def minus_source_key(row: dict[str, Any]) -> str:
    for source in source_dicts(row):
        for key in ("source_track_id", "track_id"):
            value = str(source.get(key) or "")
            if value:
                return f"track:{value}:minus"
    queue_item = str(row.get("queue_item_id") or "")
    if queue_item:
        return f"queue:{queue_item}:minus"
    source_audio_path = str(row.get("source_audio_path") or "")
    if source_audio_path:
        return Path(source_audio_path).as_posix()
    return str(row.get("artifact_relative_path") or candidate_id(row))


def load_judge_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id") or ""): row for row in iter_jsonl(path) if row.get("candidate_id")}


def minus_is_accepted(candidate_id_value: str, result_rows: dict[str, dict[str, Any]], windowed_rows: dict[str, dict[str, Any]]) -> bool:
    result = result_rows.get(candidate_id_value, {})
    windowed = windowed_rows.get(candidate_id_value, {})
    return (
        result.get("decision") == "accept_candidate"
        and result.get("minus_decision") == "accept_candidate"
        and windowed.get("decision") == "accept_candidate"
        and windowed.get("minus_decision") == "accept_candidate"
    )


def minus_sort_key(row: dict[str, Any], windowed_rows: dict[str, dict[str, Any]]) -> tuple[float, float, float, str]:
    metrics = windowed_rows.get(candidate_id(row), {})
    minus_score = float(metrics.get("minus_score") or 0.0)
    leakage = float(metrics.get("vocal_leakage_in_minus_score") or 1.0)
    mixture = float(metrics.get("mixture_consistency_score") or 0.0)
    return (minus_score, -leakage, mixture, str(row.get("artifact_relative_path") or ""))


def minus_reasons(row: dict[str, Any], windowed_rows: dict[str, dict[str, Any]]) -> list[str]:
    metrics = windowed_rows.get(candidate_id(row), {})
    return [str(reason) for reason in metrics.get("reasons") or [] if str(reason)]


def exclude_by_reasons(
    rows: Sequence[dict[str, Any]],
    *,
    windowed_rows: dict[str, dict[str, Any]],
    excluded_reasons: set[str],
) -> list[dict[str, Any]]:
    if not excluded_reasons:
        return list(rows)
    return [
        row
        for row in rows
        if not set(minus_reasons(row, windowed_rows)).intersection(excluded_reasons)
    ]


def include_by_minus_source_keys(rows: Sequence[dict[str, Any]], include_source_keys: set[str]) -> list[dict[str, Any]]:
    if not include_source_keys:
        return list(rows)
    return [row for row in rows if minus_source_key(row) in include_source_keys]


def dedupe_by_minus_source(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = minus_source_key(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def copy_selected_audio(
    *,
    source_row: dict[str, Any],
    output_dir: Path,
    group: str,
    prefix: str,
    index: int,
    route_intent: str,
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    validate_artifact_segment(group)
    source_path = Path(str(source_row.get("listen_audio_path") or ""))
    filename = indexed_audio_filename(prefix, index)
    target_path = output_dir / ARTIFACT_AUDIO_DIR / group / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    metadata = audio_metadata(target_path)
    return {
        "schema": "short_listen_repair_selection_audio.v1",
        "selection_rank": index,
        "route_intent": route_intent,
        "stem_type": str(source_row.get("stem_type") or ""),
        "artifact_alias": filename.removesuffix(".flac"),
        "artifact_filename": filename,
        "artifact_relative_path": str(target_path.relative_to(output_dir).as_posix()),
        "listen_audio_path": str(target_path),
        "listen_audio_sha256": sha256_file(target_path),
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "duration_sec": metadata["duration_sec"],
        "source_audio_path": str(source_path),
        "source_artifact_relative_path": str(source_row.get("artifact_relative_path") or ""),
        "source_candidate_id": candidate_id(source_row),
        "source_row": source_row,
        "delivery_status": "accepted",
        "delivery_allowed": True,
        "presentation_status": "accepted_for_listen_pack",
        "release_status": "listen_preview_only",
        "release_ready": False,
        "can_publish_as_result": False,
        "semantic_quality_verified": False,
        **extra_fields,
    }


def summary_text(manifest: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# short listen repair selection pack",
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
        warning = ",".join(row.get("quality_warning_reasons") or [])
        lines.append(f"- `{row['artifact_relative_path']}` `{row['route_intent']}` warning=`{warning}`")
    if manifest["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{check}`" for check in manifest["failed_checks"])
    lines.append("")
    return "\n".join(lines)


def build_short_listen_repair_selection_pack(
    *,
    repair_candidate_pack_json: Path,
    minus_result_judge_jsonl: Path,
    minus_windowed_judge_jsonl: Path,
    output_dir: Path,
    max_vocal_candidates: int = 1,
    max_minus_candidates: int = 3,
    dedupe_minus_by_track: bool = False,
    exclude_minus_reasons: Sequence[str] = (),
    include_minus_source_keys: Sequence[str] = (),
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / ARTIFACT_AUDIO_DIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    progress_path = output_dir / PROGRESS_NAME
    if progress_path.exists():
        progress_path.unlink()
    progress(progress_path, "start", repair_candidate_pack_json=str(repair_candidate_pack_json))

    source_manifest = read_json(repair_candidate_pack_json)
    failed: list[str] = collect_true_flag_failures("repair_candidate_pack", source_manifest)
    if source_manifest.get("schema") != "short_listen_repair_candidate_pack_manifest.v1":
        failed.append("repair_candidate_pack_schema_unexpected")
    if source_manifest.get("delivery_status") != "accepted" or source_manifest.get("delivery_allowed") is not True:
        failed.append("repair_candidate_pack_not_accepted")
    if source_manifest.get("failed_checks") not in ([], None):
        failed.append("repair_candidate_pack_failed_checks_not_empty")
    selected_jsonl = Path(str(source_manifest.get("selected_jsonl") or ""))
    if not selected_jsonl.exists():
        failed.append("repair_candidate_pack_selected_jsonl_missing")
    if not minus_result_judge_jsonl.exists():
        failed.append("minus_result_judge_missing")
    if not minus_windowed_judge_jsonl.exists():
        failed.append("minus_windowed_judge_missing")

    selected: list[dict[str, Any]] = []
    if not failed:
        source_rows = [row for row in iter_jsonl(selected_jsonl) if accepted_candidate_row(row)]
        result_rows = load_judge_rows(minus_result_judge_jsonl)
        windowed_rows = load_judge_rows(minus_windowed_judge_jsonl)
        vocal_rows = sorted(
            [row for row in source_rows if row.get("route_intent") == "vocal_candidate_replacement"],
            key=lambda row: (reference_sdr(row), str(row.get("artifact_relative_path") or "")),
            reverse=True,
        )[: max(0, max_vocal_candidates)]
        minus_candidates = [
            row
            for row in source_rows
            if row.get("route_intent") == "minus_repair" and minus_is_accepted(candidate_id(row), result_rows, windowed_rows)
        ]
        include_source_key_set = {str(key) for key in include_minus_source_keys if str(key)}
        minus_candidates = include_by_minus_source_keys(minus_candidates, include_source_key_set)
        excluded_reason_set = {str(reason) for reason in exclude_minus_reasons if str(reason)}
        minus_candidates = exclude_by_reasons(
            minus_candidates,
            windowed_rows=windowed_rows,
            excluded_reasons=excluded_reason_set,
        )
        ranked_minus_rows = sorted(minus_candidates, key=lambda row: minus_sort_key(row, windowed_rows), reverse=True)
        if dedupe_minus_by_track:
            ranked_minus_rows = dedupe_by_minus_source(ranked_minus_rows)
        minus_rows = ranked_minus_rows[: max(0, max_minus_candidates)]
        if max_vocal_candidates > 0 and not vocal_rows:
            failed.append("no_vocal_selection_candidates")
        if max_minus_candidates > 0 and not minus_rows:
            failed.append("no_minus_selection_candidates")
        if not failed:
            for index, row in enumerate(vocal_rows, start=1):
                selected.append(
                    copy_selected_audio(
                        source_row=row,
                        output_dir=output_dir,
                        group="voc",
                        prefix="v",
                        index=index,
                        route_intent="vocal_candidate_replacement",
                        extra_fields={
                            "selection_basis": "highest_reference_sdr",
                            "reference_metrics": row.get("reference_metrics") if isinstance(row.get("reference_metrics"), dict) else {},
                            "quality_warning_reasons": [],
                        },
                    )
                )
            for index, row in enumerate(minus_rows, start=1):
                metrics = windowed_rows.get(candidate_id(row), {})
                reasons = minus_reasons(row, windowed_rows)
                selected.append(
                    copy_selected_audio(
                        source_row=row,
                        output_dir=output_dir,
                        group="min",
                        prefix="m",
                        index=index,
                        route_intent="minus_repair",
                        extra_fields={
                            "selection_basis": "accepted_minus_windowed_judge_rank",
                            "minus_judge": {
                                "candidate_id": candidate_id(row),
                                "minus_score": metrics.get("minus_score"),
                                "vocal_leakage_in_minus_score": metrics.get("vocal_leakage_in_minus_score"),
                                "mixture_consistency_score": metrics.get("mixture_consistency_score"),
                                "decision": metrics.get("decision"),
                                "minus_decision": metrics.get("minus_decision"),
                            },
                            "quality_warning_reasons": reasons,
                            "minus_source_key": minus_source_key(row),
                        },
                    )
                )

    failed = stable_unique(failed)
    accepted = bool(selected) and not failed
    if not accepted:
        selected = []
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

    manifest = {
        "schema": "short_listen_repair_selection_pack_manifest.v1",
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
        "repair_candidate_pack_json": str(repair_candidate_pack_json),
        "minus_result_judge_jsonl": str(minus_result_judge_jsonl),
        "minus_windowed_judge_jsonl": str(minus_windowed_judge_jsonl),
        "minus_selection_policy": {
            "dedupe_minus_by_track": dedupe_minus_by_track,
            "exclude_minus_reasons": stable_unique(str(reason) for reason in exclude_minus_reasons if str(reason)),
            "include_minus_source_keys": stable_unique(str(key) for key in include_minus_source_keys if str(key)),
        },
        "failed_checks": failed,
        "selected_jsonl": str(output_dir / SELECTED_NAME),
        "summary_md": str(output_dir / SUMMARY_NAME),
        "progress_jsonl": str(progress_path),
        "artifact_name_policy": {
            "schema": "short_windows_safe_artifact_names.v1",
            "audio_dir": ARTIFACT_AUDIO_DIR,
            "group_dir_strategy": "voc|min",
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
    parser = argparse.ArgumentParser(description="Build a concise short listen selection pack from repair candidates.")
    parser.add_argument("--repair-candidate-pack-json", type=Path, required=True)
    parser.add_argument("--minus-result-judge-jsonl", type=Path, required=True)
    parser.add_argument("--minus-windowed-judge-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-vocal-candidates", type=int, default=1)
    parser.add_argument("--max-minus-candidates", type=int, default=3)
    parser.add_argument("--dedupe-minus-by-track", action="store_true")
    parser.add_argument("--exclude-minus-reason", action="append", default=[])
    parser.add_argument("--include-minus-source-key", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="build_short_listen_repair_selection_pack.py",
        entrypoint_path="/app/tools/build_short_listen_repair_selection_pack.py",
    )
    args = parse_args(argv)
    manifest = build_short_listen_repair_selection_pack(
        repair_candidate_pack_json=args.repair_candidate_pack_json,
        minus_result_judge_jsonl=args.minus_result_judge_jsonl,
        minus_windowed_judge_jsonl=args.minus_windowed_judge_jsonl,
        output_dir=args.output_dir,
        max_vocal_candidates=int(args.max_vocal_candidates),
        max_minus_candidates=int(args.max_minus_candidates),
        dedupe_minus_by_track=bool(args.dedupe_minus_by_track),
        exclude_minus_reasons=list(args.exclude_minus_reason),
        include_minus_source_keys=list(args.include_minus_source_key),
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
