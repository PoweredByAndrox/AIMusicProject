from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.instrumental_self_judge_gate import judge_instrumental


MATRIX_JSONL = "rows.jsonl"
MANIFEST_JSON = "manifest.json"
SUMMARY_MD = "summary.md"
PROGRESS_JSONL = "progress.jsonl"
EXPECTED_ACCELERATION_MODE = "cpu_dsp_self_judge_matrix"

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "release_ready": False,
}

JudgeRunner = Callable[..., dict[str, Any]]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def log_progress(path: Path, stage: str, *, status: str = "progress", **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "type": "no_reference_minus_self_judge_matrix_progress",
        "stage": stage,
        "status": status,
        "elapsed_sec": round(float(metadata.pop("elapsed_sec", 0.0)), 6),
        "metadata": metadata,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:140] or fallback


def row_id(row: dict[str, Any], index: int) -> tuple[str, str]:
    track_id = str(row.get("track_id") or Path(str(row.get("mix_path") or row.get("source_audio_path") or "")).stem)
    candidate_id = str(row.get("candidate_id") or row.get("minus_route_id") or f"candidate_{index:04d}")
    return track_id, candidate_id


def audio_paths(row: dict[str, Any]) -> dict[str, str]:
    return {
        "mix_path": str(row.get("mix_path") or row.get("source_audio_path") or ""),
        "vocal_path": str(row.get("vocal_path") or row.get("selected_vocal_path") or ""),
        "minus_path": str(row.get("instrumental_path") or row.get("minus_path") or row.get("selected_minus_path") or ""),
    }


def missing_audio_inputs(paths: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for key in ("mix_path", "minus_path"):
        value = paths.get(key, "")
        if not value or not Path(value).exists():
            missing.append(key)
    return missing


def blocked_missing_audio_row(
    *,
    row: dict[str, Any],
    index: int,
    paths: dict[str, str],
    missing: list[str],
    track_alias: str,
    candidate_alias: str,
) -> dict[str, Any]:
    track_id, candidate_id = row_id(row, index)
    return {
        "schema": "no_reference_minus_self_judge_matrix_row.v1",
        "row_index": index,
        "track_id": track_id,
        "candidate_id": candidate_id,
        "self_judge_alias": {"track_alias": track_alias, "candidate_alias": candidate_alias},
        **paths,
        "self_judge_status": "blocked_missing_audio",
        "dominant_failure_reason": "missing_audio_inputs",
        "missing_audio_inputs": missing,
        "blocker_count": None,
        "segment_count": None,
        "blocker_ratio": None,
        "issue_counts": {},
        "self_judge_manifest": "",
        "listen_candidate_eligible": False,
        "delivery_allowed": False,
        "release_ready": False,
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
    }


def matrix_row_from_manifest(
    *,
    source_row: dict[str, Any],
    index: int,
    paths: dict[str, str],
    manifest: dict[str, Any],
    manifest_path: Path,
    track_alias: str,
    candidate_alias: str,
) -> dict[str, Any]:
    track_id, candidate_id = row_id(source_row, index)
    overall = manifest.get("overall") if isinstance(manifest.get("overall"), dict) else {}
    status = str(overall.get("status") or "unknown")
    blocker_count = int(overall.get("blocker_count") or 0)
    segment_count = int(overall.get("segment_count") or 0)
    issue_counts = overall.get("issue_counts") if isinstance(overall.get("issue_counts"), dict) else {}
    eligible = status == "proxy_pass" and blocker_count == 0
    return {
        "schema": "no_reference_minus_self_judge_matrix_row.v1",
        "row_index": index,
        "track_id": track_id,
        "candidate_id": candidate_id,
        "self_judge_alias": {"track_alias": track_alias, "candidate_alias": candidate_alias},
        **paths,
        "self_judge_status": status,
        "dominant_failure_reason": "" if eligible else "self_judge_proxy_blocked",
        "missing_audio_inputs": [],
        "blocker_count": blocker_count,
        "segment_count": segment_count,
        "blocker_ratio": float(overall.get("blocker_ratio") or 0.0),
        "issue_counts": issue_counts,
        "self_judge_manifest": str(manifest_path),
        "listen_candidate_eligible": eligible,
        "delivery_allowed": False,
        "release_ready": False,
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    source_count: int,
    filtered_out_candidates: int,
    max_candidates: int,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("self_judge_status") or "unknown") for row in rows)
    return {
        "schema": "no_reference_minus_self_judge_matrix_summary.v1",
        "source_candidate_rows": source_count,
        "filtered_out_candidates": int(filtered_out_candidates),
        "evaluated_candidates": len(rows),
        "max_candidates": int(max_candidates),
        "eligible_candidates": sum(1 for row in rows if row.get("listen_candidate_eligible") is True),
        "blocked_candidates": sum(1 for row in rows if row.get("listen_candidate_eligible") is not True),
        "blocked_missing_audio": int(status_counts.get("blocked_missing_audio", 0)),
        "status_counts": dict(sorted(status_counts.items())),
        "track_ids": sorted({str(row.get("track_id") or "") for row in rows if row.get("track_id")}),
        "expected_acceleration_mode": EXPECTED_ACCELERATION_MODE,
        "gpu_inference_required": False,
        "gpu_inference_skipped": True,
        "delivery_allowed": False,
        "release_ready": False,
    }


def build_manifest(output_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_type": "no_reference_minus_self_judge_matrix",
        "schema": "no_reference_minus_self_judge_matrix_manifest.v1",
        "output_root": str(output_root),
        "output_files": {
            "matrix_jsonl": str(output_root / MATRIX_JSONL),
            "manifest_json": str(output_root / MANIFEST_JSON),
            "summary_md": str(output_root / SUMMARY_MD),
            "progress_jsonl": str(output_root / PROGRESS_JSONL),
        },
        "summary": summary,
        "eligible_candidate_ids": [
            str(row.get("candidate_id") or "") for row in rows if row.get("listen_candidate_eligible") is True
        ],
        "artifact_name_policy": {
            "path_segment_limit": 32,
            "raw_ids_in_paths": False,
            "provenance_in_manifest": True,
            "nested_artifact_aliases": "tNNN/cNNN",
        },
        "delivery_allowed": False,
        "presentation_status": "diagnostic_self_judge_matrix_not_for_listen_pack",
        "release_status": "not_release_ready",
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        **SAFETY_FLAGS,
    }


def build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# No-Reference Minus Self-Judge Matrix",
        "",
        f"- source_candidate_rows: {summary.get('source_candidate_rows')}",
        f"- evaluated_candidates: {summary.get('evaluated_candidates')}",
        f"- eligible_candidates: {summary.get('eligible_candidates')}",
        f"- blocked_candidates: {summary.get('blocked_candidates')}",
        f"- blocked_missing_audio: {summary.get('blocked_missing_audio')}",
        f"- expected_acceleration_mode: {summary.get('expected_acceleration_mode')}",
        f"- gpu_inference_skipped: {str(summary.get('gpu_inference_skipped') is True).lower()}",
        f"- delivery_allowed: {str(summary.get('delivery_allowed') is True).lower()}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        "",
        "## Status Counts",
        "",
    ]
    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), dict) else {}
    for key, value in sorted(status_counts.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run_self_judge_matrix(
    *,
    candidate_jsonl: Path,
    output_root: Path,
    track_id: str | None = None,
    minus_path_contains: str | None = None,
    segment_sec: float = 6.0,
    max_candidates: int = 0,
    judge_runner: JudgeRunner = judge_instrumental,
) -> dict[str, Any]:
    start = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / PROGRESS_JSONL
    progress_path.write_text("", encoding="utf-8")
    source_rows = list(iter_jsonl(candidate_jsonl))
    filtered_rows = []
    for row in source_rows:
        if track_id is not None and str(row.get("track_id") or "") != str(track_id):
            continue
        if minus_path_contains is not None and str(minus_path_contains) not in audio_paths(row)["minus_path"]:
            continue
        filtered_rows.append(row)
    selected_rows = filtered_rows[:max_candidates] if max_candidates > 0 else filtered_rows
    filtered_out_candidates = len(source_rows) - len(filtered_rows)
    log_progress(
        progress_path,
        "start",
        source_candidate_rows=len(source_rows),
        filtered_out_candidates=filtered_out_candidates,
        selected_candidates=len(selected_rows),
        track_id_filter=str(track_id or ""),
        minus_path_contains_filter=str(minus_path_contains or ""),
        expected_acceleration_mode=EXPECTED_ACCELERATION_MODE,
        gpu_inference_skipped=True,
        gpu_inference_skip_reason="self_judge_consumes_precomputed_audio",
    )

    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(selected_rows, start=1):
        track_id, candidate_id = row_id(source_row, index)
        track_alias = f"t{index:03d}"
        candidate_alias = f"c{index:03d}"
        paths = audio_paths(source_row)
        missing = missing_audio_inputs(paths)
        log_progress(progress_path, "candidate_start", row_index=index, track_id=track_id, candidate_id=candidate_id)
        if missing:
            row = blocked_missing_audio_row(
                row=source_row,
                index=index,
                paths=paths,
                missing=missing,
                track_alias=track_alias,
                candidate_alias=candidate_alias,
            )
            rows.append(row)
            log_progress(
                progress_path,
                "candidate_end",
                status="blocked_missing_audio",
                row_index=index,
                track_id=track_id,
                candidate_id=candidate_id,
                missing_audio_inputs=missing,
            )
            continue

        candidate_dir = output_root / track_alias / candidate_alias
        manifest = judge_runner(
            source_audio_path=Path(paths["mix_path"]),
            candidate_instrumental_path=Path(paths["minus_path"]),
            output_dir=candidate_dir,
            candidate_vocal_path=Path(paths["vocal_path"]) if paths.get("vocal_path") else None,
            candidate_label=candidate_id,
            segment_sec=float(segment_sec),
            short_output_names=True,
        )
        manifest_path = Path(str(manifest.get("manifest_path") or candidate_dir / "manifest.json"))
        row = matrix_row_from_manifest(
            source_row=source_row,
            index=index,
            paths=paths,
            manifest=manifest,
            manifest_path=manifest_path,
            track_alias=track_alias,
            candidate_alias=candidate_alias,
        )
        rows.append(row)
        log_progress(
            progress_path,
            "candidate_end",
            status=str(row["self_judge_status"]),
            row_index=index,
            track_id=track_id,
            candidate_id=candidate_id,
            blocker_count=row["blocker_count"],
            segment_count=row["segment_count"],
            eligible=row["listen_candidate_eligible"],
        )

    summary = build_summary(
        rows,
        source_count=len(source_rows),
        filtered_out_candidates=filtered_out_candidates,
        max_candidates=max_candidates,
    )
    manifest = build_manifest(output_root, rows, summary)
    write_jsonl(output_root / MATRIX_JSONL, rows)
    write_json(output_root / MANIFEST_JSON, manifest)
    (output_root / SUMMARY_MD).write_text(build_summary_md(summary), encoding="utf-8")
    log_progress(
        progress_path,
        "finish",
        status="completed",
        elapsed_sec=time.monotonic() - start,
        eligible_candidates=summary["eligible_candidates"],
        blocked_candidates=summary["blocked_candidates"],
    )
    return {"rows": rows, "summary": summary, "manifest": manifest}


def running_inside_docker() -> bool:
    return Path("/.dockerenv").exists() or Path("/app").exists()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a no-reference minus self-judge matrix over candidate rows.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--minus-path-contains")
    parser.add_argument("--segment-sec", type=float, default=6.0)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args(argv)
    if not running_inside_docker():
        raise SystemExit("Docker runtime required for no-reference minus self-judge matrix execution")
    result = run_self_judge_matrix(
        candidate_jsonl=args.candidate_jsonl,
        output_root=args.output_root,
        track_id=str(args.track_id) if args.track_id else None,
        minus_path_contains=str(args.minus_path_contains) if args.minus_path_contains else None,
        segment_sec=float(args.segment_sec),
        max_candidates=int(args.max_candidates),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
