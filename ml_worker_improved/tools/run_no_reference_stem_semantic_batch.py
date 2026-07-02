from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_stem_candidate_score_matrix import (  # noqa: E402
    build_stem_candidate_score_matrix,
    write_outputs as write_matrix_outputs,
)
from tools.artifact_status_contract import (  # noqa: E402
    SEMANTIC_JUDGE_INPUT_DELIVERY_STATUS,
    SEMANTIC_JUDGE_INPUT_PRESENTATION_STATUS,
    SEMANTIC_JUDGE_INPUT_RELEASE_STATUS,
)
from tools.run_no_reference_stem_semantic_judge import (  # noqa: E402
    SAFETY_FLAGS,
    run_stem_semantic_judge,
    write_outputs as write_judge_outputs,
)


BATCH_SUMMARY_JSON_NAME = "no_reference_stem_semantic_batch_summary.json"
BATCH_SUMMARY_MD_NAME = "no_reference_stem_semantic_batch_summary.md"
BATCH_PROGRESS_JSONL_NAME = "no_reference_stem_semantic_batch_progress.jsonl"
SHORT_BATCH_SUMMARY_JSON_NAME = "summary.json"
SHORT_BATCH_SUMMARY_MD_NAME = "summary.md"
SHORT_BATCH_PROGRESS_JSONL_NAME = "progress.jsonl"
SHORT_JUDGE_JSONL_NAME = "rows.jsonl"
SHORT_JUDGE_SUMMARY_JSON_NAME = "summary.json"
SHORT_JUDGE_SUMMARY_MD_NAME = "summary.md"
SHORT_MATRIX_JSON_NAME = "matrix.json"
SHORT_MATRIX_JSONL_NAME = "rows.jsonl"
SHORT_MATRIX_SUMMARY_NAME = "summary.md"
REVIEW_JSON_NAME = "review.json"
REVIEW_ROWS_NAME = "rows.jsonl"
REVIEW_SUMMARY_NAME = "summary.md"
REVIEW_QUEUE_DELIVERY_STATUS = "semantic_review_pending"
REVIEW_QUEUE_PRESENTATION_STATUS = "semantic_review_queue_not_for_listen_pack"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def safe_dir_name(value: str) -> str:
    text = str(value or "unknown_track").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown_track"


def short_segment(value: str, *, fallback: str, max_len: int = 24) -> str:
    text = re.sub(r"[^a-z0-9._-]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        text = fallback
    return text[:max_len] or fallback


def stable_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def discover_listen_manifests(roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "no_reference_stem_listen_preview_manifest.json":
            paths.append(root)
        elif root.exists():
            paths.extend(sorted(root.rglob("no_reference_stem_listen_preview_manifest.json")))
    return sorted(dict.fromkeys(paths))


def group_manifests_by_track(paths: Sequence[Path], track_ids: Sequence[str] | None = None) -> dict[str, list[Path]]:
    allowed = {str(track_id) for track_id in track_ids or [] if str(track_id)}
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        payload = read_json(path)
        track_id = str(payload.get("track_id") or "unknown_track")
        if allowed and track_id not in allowed:
            continue
        grouped.setdefault(track_id, []).append(path)
    return {track_id: sorted(items) for track_id, items in sorted(grouped.items())}


def pack_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("listen_audio_files"), list):
        return [row for row in payload["listen_audio_files"] if isinstance(row, dict)]
    if isinstance(payload.get("items"), list):
        return [row for row in payload["items"] if isinstance(row, dict)]
    return []


def validate_input_pack(payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    schema = str(payload.get("schema") or "")
    if schema not in {
        "music_intelligence_stem_listen_pack_manifest.v1",
        "aggregate_short_listen_pack_manifest.v1",
    }:
        failed.append("input_pack_schema_unexpected")
    elif schema == "music_intelligence_stem_listen_pack_manifest.v1":
        if payload.get("review_queue_ready") is not True:
            failed.append("input_pack_review_queue_not_ready")
        if payload.get("delivery_status") != REVIEW_QUEUE_DELIVERY_STATUS:
            failed.append("input_pack_delivery_status_not_semantic_review_pending")
        if payload.get("delivery_allowed") is not False:
            failed.append("input_pack_delivery_allowed_not_false")
        if payload.get("presentation_status") != REVIEW_QUEUE_PRESENTATION_STATUS:
            failed.append("input_pack_presentation_status_not_review_queue")
        if payload.get("listenable") is not False:
            failed.append("input_pack_listenable_not_false")
    elif schema == "aggregate_short_listen_pack_manifest.v1":
        if payload.get("delivery_status") != "accepted":
            failed.append("input_pack_delivery_status_not_accepted")
        if payload.get("delivery_allowed") is not True:
            failed.append("input_pack_delivery_allowed_not_true")
        if payload.get("presentation_status") != "accepted_for_listen_pack":
            failed.append("input_pack_presentation_status_not_accepted_for_listen_pack")
    if payload.get("release_ready") is True:
        failed.append("input_pack_release_ready_true")
    if payload.get("can_publish_as_result") is True:
        failed.append("input_pack_can_publish_as_result_true")
    if payload.get("failed_checks") not in ([], None):
        failed.append("input_pack_failed_checks_not_empty")
    return failed


def validate_music_intelligence_queue_row(row: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if row.get("delivery_status") != REVIEW_QUEUE_DELIVERY_STATUS:
        failed.append("input_pack_item_delivery_status_not_semantic_review_pending")
    if row.get("delivery_allowed") is not False:
        failed.append("input_pack_item_delivery_allowed_not_false")
    if row.get("presentation_status") != REVIEW_QUEUE_PRESENTATION_STATUS:
        failed.append("input_pack_item_presentation_status_not_review_queue")
    return failed


def resolve_pack_audio_path(row: dict[str, Any], pack_path: Path) -> str:
    raw = str(row.get("listen_audio_path") or row.get("audio_path") or row.get("source_audio_path") or "")
    path = Path(raw)
    if not path.is_absolute():
        path = pack_path.parent / path
    return str(path)


def resolve_pack_path(raw: Any, base_path: Path | None = None) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        return path
    if not path.is_absolute() and base_path is not None:
        path = base_path.parent / path
    return path


def existing_path_text(raw: Any, base_path: Path | None = None) -> str:
    path = resolve_pack_path(raw, base_path)
    return str(path) if str(path) and path.exists() and path.is_file() else ""


def row_group_key(row: dict[str, Any]) -> str:
    return str(row.get("track_alias") or row.get("source_alias") or row.get("track_id") or "")


def row_source_dicts(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = [row]
    source_row = row.get("source_row")
    if isinstance(source_row, dict):
        result.append(source_row)
    return result


def direct_mix_path_from_row(row: dict[str, Any], pack_path: Path) -> str:
    for source in row_source_dicts(row):
        for key in ("mix_path", "source_mix_path", "input_audio_path", "raw_audio_path"):
            if key not in source:
                continue
            text = existing_path_text(source.get(key), pack_path)
            if text:
                return text
    return ""


def read_manifest_mix_path(manifest_path: Path) -> str:
    if not manifest_path.exists() or not manifest_path.is_file():
        return ""
    try:
        payload = read_json(manifest_path)
    except Exception:
        return ""
    stem_manifest_json = payload.get("stem_manifest_json")
    if stem_manifest_json:
        nested = resolve_pack_path(stem_manifest_json, manifest_path)
        text = read_manifest_mix_path(nested)
        if text:
            return text
    for key in ("mix_path", "source_mix_path", "input_audio_path", "raw_audio_path"):
        text = existing_path_text(payload.get(key), manifest_path)
        if text:
            return text
    if isinstance(payload.get("stems"), dict):
        for key in ("audio_path", "source_audio_path"):
            text = existing_path_text(payload.get(key), manifest_path)
            if text:
                return text
    return ""


def manifest_mix_path_from_row(row: dict[str, Any], pack_path: Path) -> str:
    for source in row_source_dicts(row):
        for key in ("stem_manifest_json", "source_stem_manifest_json", "source_manifest", "upstream_source_manifest"):
            raw = source.get(key)
            if not raw:
                continue
            text = read_manifest_mix_path(resolve_pack_path(raw, pack_path))
            if text:
                return text
    return ""


def resolve_pair_mix_path(rows: Sequence[dict[str, Any]], pack_path: Path) -> str:
    for row in rows:
        text = direct_mix_path_from_row(row, pack_path)
        if text:
            return text
    for row in rows:
        text = manifest_mix_path_from_row(row, pack_path)
        if text:
            return text
    return ""


def mode_specific_candidate_rows(payload: dict[str, Any], pack_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pack_rows(payload):
        grouped.setdefault(row_group_key(row), []).append(row)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for group, rows in grouped.items():
        vocal_row = next((row for row in rows if str(row.get("stem_type") or "").lower() in {"vocal", "vocals"}), None)
        minus_row = next((row for row in rows if str(row.get("stem_type") or "").lower() in {"minus", "instrumental"}), None)
        if vocal_row is None or minus_row is None:
            continue
        mix_path = resolve_pair_mix_path(rows, pack_path)
        vocal_path = resolve_pack_audio_path(vocal_row, pack_path)
        minus_path = resolve_pack_audio_path(minus_row, pack_path)
        if not mix_path or not Path(vocal_path).exists() or not Path(minus_path).exists():
            continue
        candidate = {
            "track_id": short_segment(group, fallback="t000", max_len=16),
            "candidate_id": f"{short_segment(group, fallback='t000', max_len=12)}_pair",
            "mix_path": mix_path,
            "vocal_path": vocal_path,
            "instrumental_path": minus_path,
            "minus_path": minus_path,
            "evidence_source": "short_pack_sibling_vocal_minus",
        }
        result[(group, "vocal")] = candidate
        result[(group, "minus")] = candidate
    return result


def materialized_manifest_payload(
    row: dict[str, Any],
    pack_path: Path,
    *,
    mode_specific_candidate_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    track_alias = str(row.get("track_alias") or row.get("source_alias") or "")
    stem_type = str(row.get("stem_type") or "unknown")
    payload = {
        "manifest_type": "no_reference_stem_listen_preview",
        "delivery_status": SEMANTIC_JUDGE_INPUT_DELIVERY_STATUS,
        "delivery_allowed": False,
        "presentation_status": SEMANTIC_JUDGE_INPUT_PRESENTATION_STATUS,
        "release_status": SEMANTIC_JUDGE_INPUT_RELEASE_STATUS,
        "semantic_judge_input": True,
        "stem_type": stem_type,
        "track_id": track_alias,
        "source_track_id": str(row.get("source_track_id") or row.get("track_id") or ""),
        "source_stem_id": str(row.get("source_stem_id") or row.get("stem_id") or ""),
        "model_name": str(row.get("model_name") or "short_pack_materialized_view"),
        "selected_audio_path": resolve_pack_audio_path(row, pack_path),
        "audio_integrity_gate": {
            "status": "passed",
            "duration_sec": row.get("duration_sec") or 0.0,
            "rms": row.get("rms") or 0.0,
            "peak": row.get("peak") or 0.0,
        },
        "gpu_provenance_gate": {
            "status": "passed",
            "gpu_inference_required": False,
            "gpu_inference_skipped": True,
            "expected_acceleration_mode": "cpu_dsp_semantic_pack_bridge",
            "cpu_dsp_only_route": True,
            "cuda_available": False,
            "resolved_device": "cpu",
        },
        "source_contract_gate": {"status": "passed"},
        "source_alias": str(row.get("source_alias") or row.get("track_alias") or ""),
        "source_pack_json": str(pack_path),
        "source_manifest": str(row.get("source_manifest") or ""),
        "upstream_source_manifest": str(row.get("upstream_source_manifest") or ""),
        "source_pack_audio_path": str(row.get("artifact_relative_path") or row.get("audio_path") or ""),
        "semantic_quality_verified": False,
        "listenable": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "can_publish_as_result": False,
        "safety": dict(SAFETY_FLAGS),
    }
    source_row = row.get("source_row")
    if isinstance(source_row, dict):
        payload["source_row"] = source_row
    if mode_specific_candidate_row is not None and stem_type.lower() in {"vocal", "minus"}:
        payload["mode_specific_candidate_row"] = dict(mode_specific_candidate_row)
    return payload


def materialize_input_pack_manifests(input_pack_jsons: Sequence[Path], output_root: Path) -> tuple[list[Path], list[str]]:
    source_root = output_root / "src"
    if source_root.exists():
        shutil.rmtree(source_root)
    paths: list[Path] = []
    failed: list[str] = []
    aliases: dict[str, str] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}
    for pack_path in input_pack_jsons:
        payload = read_json(pack_path)
        pack_failed = validate_input_pack(payload)
        failed.extend(pack_failed)
        if pack_failed:
            continue
        mode_specific_rows = mode_specific_candidate_rows(payload, pack_path)
        for row in pack_rows(payload):
            if payload.get("schema") == "music_intelligence_stem_listen_pack_manifest.v1":
                row_failed = validate_music_intelligence_queue_row(row)
                failed.extend(row_failed)
                if row_failed:
                    continue
            if row.get("release_ready") is True:
                failed.append("input_pack_item_release_ready_true")
                continue
            if row.get("can_publish_as_result") is True:
                failed.append("input_pack_item_can_publish_as_result_true")
                continue
            raw_alias = str(row.get("track_alias") or row.get("source_alias") or row.get("track_id") or "")
            track_key = raw_alias or f"track_{len(aliases) + 1}"
            track_alias = aliases.setdefault(track_key, short_segment(raw_alias, fallback=f"t{len(aliases) + 1:03d}", max_len=16))
            stem_type = short_segment(str(row.get("stem_type") or "unknown"), fallback="stem", max_len=16)
            duplicate_key = (track_alias, stem_type)
            duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
            suffix = "" if duplicate_counts[duplicate_key] == 1 else f"_{duplicate_counts[duplicate_key]:03d}"
            manifest_path = source_root / track_alias / f"{stem_type}{suffix}.json"
            materialized = materialized_manifest_payload(
                row,
                pack_path,
                mode_specific_candidate_row=mode_specific_rows.get((row_group_key(row), str(row.get("stem_type") or "").lower())),
            )
            materialized["track_id"] = track_alias
            write_json(manifest_path, materialized)
            paths.append(manifest_path)
    return sorted(paths), stable_unique(failed)


def aggregate_safety(track_summaries: Sequence[dict[str, Any]]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for track in track_summaries:
        track_safety = track.get("safety")
        if not isinstance(track_safety, dict):
            continue
        for key in safety:
            safety[key] = bool(safety[key] or track_safety.get(key) is True)
    return safety


def batch_summary_names(short_output_names: bool) -> tuple[str, str, str]:
    if short_output_names:
        return SHORT_BATCH_SUMMARY_JSON_NAME, SHORT_BATCH_SUMMARY_MD_NAME, SHORT_BATCH_PROGRESS_JSONL_NAME
    return BATCH_SUMMARY_JSON_NAME, BATCH_SUMMARY_MD_NAME, BATCH_PROGRESS_JSONL_NAME


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Stem Semantic Batch Summary",
        "",
        f"- batch_status: {summary.get('batch_status')}",
        f"- track_count: {summary.get('track_count')}",
        f"- candidate_count: {summary.get('candidate_count')}",
        f"- accepted_candidates: {summary.get('accepted_candidates')}",
        f"- blocked_hard_case: {summary.get('blocked_hard_case')}",
        f"- semantic_quality_verified_count: {summary.get('semantic_quality_verified_count')}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        f"- acceleration_mode: {summary.get('acceleration_mode')}",
        f"- gpu_inference_skipped: {str(summary.get('gpu_inference_skipped') is True).lower()}",
        "",
        "## Tracks",
        "",
    ]
    for track in summary.get("tracks", []):
        lines.append(
            "- "
            f"{track.get('track_id')} "
            f"candidates={track.get('candidate_count')} "
            f"accepted={track.get('accepted_candidates')} "
            f"blocked={track.get('blocked_hard_case')} "
            f"matrix={track.get('matrix_json')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_short_judge_outputs(output_root: Path, result: dict[str, Any], source_paths: Sequence[Path], *, elapsed_sec: float) -> Path:
    rows = result["rows"]
    summary = dict(result["summary"])
    summary["source_listen_manifests"] = [{"path": str(path)} for path in source_paths]
    summary["elapsed_sec"] = round(elapsed_sec, 6)
    write_jsonl(output_root / SHORT_JUDGE_JSONL_NAME, rows)
    write_json(output_root / SHORT_JUDGE_SUMMARY_JSON_NAME, summary)
    lines = [
        "# stem semantic judge",
        "",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        f"- accepted_candidates: `{summary.get('accepted_candidates')}`",
        f"- blocked_hard_case: `{summary.get('blocked_hard_case')}`",
        f"- release_ready: `{str(summary.get('release_ready') is True).lower()}`",
        "",
    ]
    (output_root / SHORT_JUDGE_SUMMARY_MD_NAME).write_text("\n".join(lines), encoding="utf-8")
    return output_root / SHORT_JUDGE_JSONL_NAME


def write_short_matrix_outputs(output_root: Path, matrix: dict[str, Any]) -> Path:
    write_json(output_root / SHORT_MATRIX_JSON_NAME, matrix)
    write_jsonl(output_root / SHORT_MATRIX_JSONL_NAME, matrix.get("ranked_candidates", []))
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    lines = [
        "# stem score matrix",
        "",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        f"- accepted_listen_preview_count: `{summary.get('accepted_listen_preview_count')}`",
        f"- semantic_judge_blocked_count: `{summary.get('semantic_judge_blocked_count')}`",
        f"- release_ready: `{str(summary.get('release_ready') is True).lower()}`",
        "",
    ]
    (output_root / SHORT_MATRIX_SUMMARY_NAME).write_text("\n".join(lines), encoding="utf-8")
    return output_root / SHORT_MATRIX_JSON_NAME


def review_row(row: dict[str, Any]) -> dict[str, Any]:
    blockers = [str(item) for item in row.get("stem_semantic_blockers") or [] if str(item)]
    verified = row.get("semantic_quality_verified") is True
    return {
        "schema": "semantic_stem_quality_review_row.v1",
        "group_alias": str(row.get("track_id") or ""),
        "stem_type": str(row.get("stem_type") or "unknown"),
        "audio_path": str(row.get("selected_audio_path") or ""),
        "semantic_review_required": True,
        "semantic_quality_verified": verified,
        "semantic_quality_status": "verified" if verified else "blocked",
        "review_status": "accepted" if verified else "blocked",
        "blockers": blockers,
        "stem_semantic_score": row.get("stem_semantic_score"),
        "evidence_paths": [str(row.get("source_listen_manifest", {}).get("path") or "")],
        "reviewed_at_utc": "",
        "release_ready": False,
        "can_publish_as_result": False,
    }


def review_failed_checks(rows: Sequence[dict[str, Any]], input_failed: Sequence[str]) -> list[str]:
    failed: list[str] = list(input_failed)
    for row in rows:
        if row.get("semantic_quality_verified") is True:
            continue
        group = str(row.get("track_id") or "unknown")
        stem = str(row.get("stem_type") or "unknown")
        blockers = [str(item) for item in row.get("stem_semantic_blockers") or [] if str(item)] or ["semantic_quality_not_verified"]
        for blocker in blockers:
            failed.append(f"stem_semantic_blocked:{group}:{stem}:{blocker}")
    return stable_unique(failed)


def write_semantic_review(output_root: Path, rows: Sequence[dict[str, Any]], input_pack_jsons: Sequence[Path], input_failed: Sequence[str]) -> Path:
    review_dir = output_root / "review"
    review_rows = [review_row(row) for row in rows]
    failed = review_failed_checks(rows, input_failed)
    verified = sum(1 for row in review_rows if row["semantic_quality_verified"] is True)
    manifest = {
        "schema": "semantic_stem_quality_review.v1",
        "review_status": "completed" if rows and not input_failed else "blocked",
        "reviewer_kind": "machine_proxy_semantic_batch",
        "review_tool": "run_no_reference_stem_semantic_batch.py",
        "reviewed_at_utc": "",
        "input_pack_jsons": [str(path) for path in input_pack_jsons],
        "rows_jsonl": str(review_dir / REVIEW_ROWS_NAME),
        "summary_md": str(review_dir / REVIEW_SUMMARY_NAME),
        "semantic_quality_verified_item_count": verified,
        "pending_item_count": 0,
        "total_item_count": len(review_rows),
        "reviewed_group_aliases": sorted({str(row["group_alias"]) for row in review_rows}),
        "failed_checks": failed,
        "release_ready": False,
        "can_publish_as_result": False,
        "delivery_allowed": False,
        "presentation_status": "semantic_review_artifact_not_listen_result",
        "not_claimed": ["release_ready", "publishable_result", "semantic_clean_ground_truth"],
    }
    write_jsonl(review_dir / REVIEW_ROWS_NAME, review_rows)
    write_json(review_dir / REVIEW_JSON_NAME, manifest)
    lines = [
        "# semantic stem quality review",
        "",
        f"- review_status: `{manifest['review_status']}`",
        f"- total_item_count: `{manifest['total_item_count']}`",
        f"- semantic_quality_verified_item_count: `{manifest['semantic_quality_verified_item_count']}`",
        f"- failed_check_count: `{len(failed)}`",
        f"- release_ready: `{str(manifest['release_ready']).lower()}`",
        "",
    ]
    (review_dir / REVIEW_SUMMARY_NAME).write_text("\n".join(lines), encoding="utf-8")
    return review_dir / REVIEW_JSON_NAME


def run_batch(
    *,
    listen_manifest_roots: Sequence[Path],
    input_pack_jsons: Sequence[Path] = (),
    output_root: Path,
    track_ids: Sequence[str] | None = None,
    short_output_names: bool | None = None,
) -> dict[str, Any]:
    started = time.time()
    short_names = bool(input_pack_jsons) if short_output_names is None else bool(short_output_names)
    summary_json_name, summary_md_name, progress_jsonl_name = batch_summary_names(short_names)
    progress_path = output_root / progress_jsonl_name
    output_root.mkdir(parents=True, exist_ok=True)
    materialized_manifests, input_failed = materialize_input_pack_manifests(input_pack_jsons, output_root) if input_pack_jsons else ([], [])
    manifests = discover_listen_manifests(listen_manifest_roots) + materialized_manifests
    grouped = group_manifests_by_track(manifests, track_ids)
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_stem_semantic_batch_start",
            "candidate_count": sum(len(items) for items in grouped.values()),
            "track_count": len(grouped),
            "input_pack_jsons": [str(path) for path in input_pack_jsons],
            "short_output_names": short_names,
            "failed_checks": input_failed,
            "acceleration_mode": "external_only_cpu_dsp_scoring",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "stem_semantic_batch_reads_existing_audio_only",
        },
    )

    track_summaries: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for track_index, (track_id, track_manifests) in enumerate(grouped.items(), start=1):
        track_dir = output_root / safe_dir_name(track_id)
        judge_dir = track_dir / ("judge" if short_names else "stem_semantic_judge")
        matrix_dir = track_dir / ("matrix" if short_names else "stem_score_matrix")
        append_jsonl(
            progress_path,
            {
                "event": "no_reference_stem_semantic_batch_track_start",
                "track_index": track_index,
                "track_count": len(grouped),
                "track_id": track_id,
                "candidate_count": len(track_manifests),
            },
        )
        judge_started = time.time()
        judge_result = run_stem_semantic_judge(track_manifests)
        semantic_rows.extend(judge_result["rows"])
        if short_names:
            judge_jsonl = write_short_judge_outputs(judge_dir, judge_result, track_manifests, elapsed_sec=time.time() - judge_started)
        else:
            write_judge_outputs(judge_dir, judge_result, track_manifests, elapsed_sec=time.time() - judge_started)
            judge_jsonl = judge_dir / "no_reference_stem_semantic_judge.jsonl"
        matrix = build_stem_candidate_score_matrix(track_manifests, semantic_judge_jsonl=judge_jsonl)
        if short_names:
            matrix_json = write_short_matrix_outputs(matrix_dir, matrix)
        else:
            write_matrix_outputs(matrix_dir, matrix)
            matrix_json = matrix_dir / "no_reference_stem_candidate_score_matrix.json"
        judge_summary = judge_result["summary"]
        matrix_summary = matrix["summary"]
        track_summary = {
            "track_id": track_id,
            "candidate_count": int(judge_summary.get("candidate_count") or 0),
            "accepted_candidates": int(judge_summary.get("accepted_candidates") or 0),
            "blocked_hard_case": int(judge_summary.get("blocked_hard_case") or 0),
            "semantic_quality_verified_count": int(judge_summary.get("semantic_quality_verified_count") or 0),
            "matrix_semantic_judge_blocked_count": int(matrix_summary.get("semantic_judge_blocked_count") or 0),
            "release_ready": False,
            "judge_jsonl": str(judge_jsonl),
            "judge_summary_json": str(judge_dir / (SHORT_JUDGE_SUMMARY_JSON_NAME if short_names else "no_reference_stem_semantic_judge_summary.json")),
            "matrix_json": str(matrix_json),
            "safety": judge_summary.get("safety") if isinstance(judge_summary.get("safety"), dict) else dict(SAFETY_FLAGS),
        }
        track_summaries.append(track_summary)
        append_jsonl(
            progress_path,
            {
                "event": "no_reference_stem_semantic_batch_track_end",
                "track_index": track_index,
                "track_count": len(grouped),
                "track_id": track_id,
                "candidate_count": track_summary["candidate_count"],
                "accepted_candidates": track_summary["accepted_candidates"],
                "blocked_hard_case": track_summary["blocked_hard_case"],
            },
        )

    candidate_count = sum(int(track.get("candidate_count") or 0) for track in track_summaries)
    accepted_count = sum(int(track.get("accepted_candidates") or 0) for track in track_summaries)
    blocked_count = sum(int(track.get("blocked_hard_case") or 0) for track in track_summaries)
    verified_count = sum(int(track.get("semantic_quality_verified_count") or 0) for track in track_summaries)
    if input_failed and not track_summaries:
        batch_status = "blocked_input_pack_failed"
    else:
        batch_status = "completed" if track_summaries else "blocked_no_selected_manifests"
    review_json = write_semantic_review(output_root, semantic_rows, input_pack_jsons, input_failed) if short_names else None
    summary = {
        "batch_status": batch_status,
        "track_count": len(track_summaries),
        "candidate_count": candidate_count,
        "accepted_candidates": accepted_count,
        "blocked_hard_case": blocked_count,
        "semantic_quality_verified_count": verified_count,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "failed_checks": input_failed,
        "input_pack_jsons": [str(path) for path in input_pack_jsons],
        "short_output_names": short_names,
        "semantic_review_json": str(review_json or ""),
        "acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "stem_semantic_batch_reads_existing_audio_only",
        "elapsed_sec": round(time.time() - started, 6),
        "tracks": track_summaries,
        "safety": aggregate_safety(track_summaries),
    }
    write_json(output_root / summary_json_name, summary)
    write_summary_md(output_root / summary_md_name, summary)
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_stem_semantic_batch_end",
            "batch_status": summary["batch_status"],
            "track_count": summary["track_count"],
            "candidate_count": summary["candidate_count"],
            "elapsed_sec": summary["elapsed_sec"],
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run corpus-scale no-reference non-vocal stem semantic judge and matrix.")
    parser.add_argument("--listen-manifest-root", type=Path, action="append", default=[])
    parser.add_argument("--input-pack-json", type=Path, action="append", default=[])
    parser.add_argument("--track-id", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--short-output-names", action="store_true")
    args = parser.parse_args(argv)

    summary = run_batch(
        listen_manifest_roots=args.listen_manifest_root,
        input_pack_jsons=args.input_pack_json,
        output_root=args.output_root,
        track_ids=args.track_id,
        short_output_names=args.short_output_names or bool(args.input_pack_json),
    )
    print(
        json.dumps(
            {
                "batch_status": summary["batch_status"],
                "track_count": summary["track_count"],
                "candidate_count": summary["candidate_count"],
                "accepted_candidates": summary["accepted_candidates"],
                "blocked_hard_case": summary["blocked_hard_case"],
                "semantic_quality_verified_count": summary["semantic_quality_verified_count"],
                "release_ready": summary["release_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
