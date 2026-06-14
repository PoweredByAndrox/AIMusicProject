from __future__ import annotations

import argparse
import json
import re
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
from tools.run_no_reference_stem_semantic_judge import (  # noqa: E402
    SAFETY_FLAGS,
    run_stem_semantic_judge,
    write_outputs as write_judge_outputs,
)


BATCH_SUMMARY_JSON_NAME = "no_reference_stem_semantic_batch_summary.json"
BATCH_SUMMARY_MD_NAME = "no_reference_stem_semantic_batch_summary.md"
BATCH_PROGRESS_JSONL_NAME = "no_reference_stem_semantic_batch_progress.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def safe_dir_name(value: str) -> str:
    text = str(value or "unknown_track").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown_track"


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


def aggregate_safety(track_summaries: Sequence[dict[str, Any]]) -> dict[str, bool]:
    safety = dict(SAFETY_FLAGS)
    for track in track_summaries:
        track_safety = track.get("safety")
        if not isinstance(track_safety, dict):
            continue
        for key in safety:
            safety[key] = bool(safety[key] or track_safety.get(key) is True)
    return safety


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


def run_batch(
    *,
    listen_manifest_roots: Sequence[Path],
    output_root: Path,
    track_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    progress_path = output_root / BATCH_PROGRESS_JSONL_NAME
    manifests = discover_listen_manifests(listen_manifest_roots)
    grouped = group_manifests_by_track(manifests, track_ids)
    output_root.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_stem_semantic_batch_start",
            "candidate_count": sum(len(items) for items in grouped.values()),
            "track_count": len(grouped),
            "acceleration_mode": "external_only_cpu_dsp_scoring",
            "gpu_inference_skipped": True,
            "gpu_inference_skip_reason": "stem_semantic_batch_reads_existing_audio_only",
        },
    )

    track_summaries: list[dict[str, Any]] = []
    for track_index, (track_id, track_manifests) in enumerate(grouped.items(), start=1):
        track_dir = output_root / safe_dir_name(track_id)
        judge_dir = track_dir / "stem_semantic_judge"
        matrix_dir = track_dir / "stem_score_matrix"
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
        write_judge_outputs(judge_dir, judge_result, track_manifests, elapsed_sec=time.time() - judge_started)
        judge_jsonl = judge_dir / "no_reference_stem_semantic_judge.jsonl"
        matrix = build_stem_candidate_score_matrix(track_manifests, semantic_judge_jsonl=judge_jsonl)
        write_matrix_outputs(matrix_dir, matrix)
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
            "judge_summary_json": str(judge_dir / "no_reference_stem_semantic_judge_summary.json"),
            "matrix_json": str(matrix_dir / "no_reference_stem_candidate_score_matrix.json"),
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
    summary = {
        "batch_status": "completed" if track_summaries else "blocked_no_selected_manifests",
        "track_count": len(track_summaries),
        "candidate_count": candidate_count,
        "accepted_candidates": accepted_count,
        "blocked_hard_case": blocked_count,
        "semantic_quality_verified_count": verified_count,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "acceleration_mode": "external_only_cpu_dsp_scoring",
        "gpu_inference_skipped": True,
        "gpu_inference_skip_reason": "stem_semantic_batch_reads_existing_audio_only",
        "elapsed_sec": round(time.time() - started, 6),
        "tracks": track_summaries,
        "safety": aggregate_safety(track_summaries),
    }
    write_json(output_root / BATCH_SUMMARY_JSON_NAME, summary)
    write_summary_md(output_root / BATCH_SUMMARY_MD_NAME, summary)
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
    parser.add_argument("--track-id", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = run_batch(
        listen_manifest_roots=args.listen_manifest_root,
        output_root=args.output_root,
        track_ids=args.track_id,
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
