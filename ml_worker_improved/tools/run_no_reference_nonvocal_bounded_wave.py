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

from tools.build_no_reference_stem_listen_preview_manifest import (  # noqa: E402
    build_manifest as build_listen_preview_manifest,
    build_summary as build_listen_preview_summary,
    write_json as write_listen_preview_json,
)
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.run_no_reference_stem_semantic_batch import run_batch as run_semantic_batch  # noqa: E402
from tools.run_testing_stuff_music_corpus_job_matrix import execute_job_matrix, iter_jsonl  # noqa: E402


SUMMARY_JSON_NAME = "no_reference_nonvocal_bounded_wave_summary.json"
SUMMARY_MD_NAME = "no_reference_nonvocal_bounded_wave_summary.md"
PROGRESS_JSONL_NAME = "no_reference_nonvocal_bounded_wave_progress.jsonl"
TARGET_STEMS = ("bass", "drums", "melody", "other")

SAFETY_FLAGS = {
    "reference_required": False,
    "human_csv_required": False,
    "training_started": False,
    "fine_tune_started": False,
    "models_changed": False,
    "model_registry_edited": False,
    "cleanup_routes_added": False,
    "thresholds_relaxed": False,
    "diagnostic_audio_exported": False,
    "audio_copied": False,
    "audio_exported": False,
    "release_ready": False,
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def discover_existing_listen_manifests(roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "no_reference_stem_listen_preview_manifest.json":
            paths.append(root)
        elif root.exists():
            paths.extend(sorted(root.rglob("no_reference_stem_listen_preview_manifest.json")))
    return sorted(dict.fromkeys(paths))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def existing_track_ids(roots: Sequence[Path]) -> set[str]:
    track_ids: set[str] = set()
    for path in discover_existing_listen_manifests(roots):
        try:
            payload = read_json(path)
        except Exception:
            continue
        track_id = str(payload.get("track_id") or "")
        if track_id:
            track_ids.add(track_id)
    return track_ids


def jobs_by_track(job_matrix_jsonl: Path, target_stems: Sequence[str] = TARGET_STEMS) -> dict[str, set[str]]:
    required = set(target_stems)
    grouped: dict[str, set[str]] = {}
    for row in iter_jsonl(job_matrix_jsonl):
        if row.get("execution_ready") is not True and str(row.get("job_status") or "") not in {"queued", "execution_ready"}:
            continue
        track_id = str(row.get("track_id") or "")
        stem = str(row.get("target_stem") or row.get("stem") or "")
        if track_id and stem in required:
            grouped.setdefault(track_id, set()).add(stem)
    return grouped


def select_next_track_ids(
    *,
    job_matrix_jsonl: Path,
    existing_listen_roots: Sequence[Path],
    max_tracks: int,
    target_stems: Sequence[str] = TARGET_STEMS,
) -> list[str]:
    if max_tracks <= 0:
        raise ValueError("--max-tracks must be positive")
    required = set(target_stems)
    already_evaluated = existing_track_ids(existing_listen_roots)
    grouped = jobs_by_track(job_matrix_jsonl, target_stems=target_stems)
    eligible = [
        track_id
        for track_id, stems in sorted(grouped.items())
        if track_id not in already_evaluated and required.issubset(stems)
    ]
    return eligible[:max_tracks]


def candidate_manifest_path(row: dict[str, Any]) -> Path | None:
    candidate = row.get("candidate_manifest")
    if isinstance(candidate, dict):
        path = str(candidate.get("path") or "")
        if path:
            return Path(path)
    path = str(row.get("expected_candidate_manifest") or "")
    return Path(path) if path else None


def target_stem(row: dict[str, Any]) -> str:
    candidate = row.get("candidate_manifest")
    if isinstance(candidate, dict) and str(candidate.get("target_stem") or ""):
        return str(candidate.get("target_stem"))
    return str(row.get("target_stem") or row.get("stem") or "")


def build_listen_previews_from_execution_rows(
    *,
    rows: Sequence[dict[str, Any]],
    output_root: Path,
) -> list[Path]:
    manifests: list[Path] = []
    for row in rows:
        if str(row.get("execution_status") or "") != "completed":
            continue
        manifest_path = candidate_manifest_path(row)
        stem = target_stem(row)
        track_id = str(row.get("track_id") or "")
        if manifest_path is None or not stem or not track_id:
            continue
        listen_manifest = build_listen_preview_manifest(
            stem_manifest_json=manifest_path,
            target_stem=stem,
            track_id=track_id,
        )
        out_dir = output_root / track_id / stem
        out_manifest = out_dir / "no_reference_stem_listen_preview_manifest.json"
        out_summary = out_dir / "no_reference_stem_listen_preview_summary.md"
        write_listen_preview_json(out_manifest, listen_manifest)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(build_listen_preview_summary(listen_manifest), encoding="utf-8")
        manifests.append(out_manifest)
    return manifests


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Non-Vocal Bounded Wave Summary",
        "",
        f"- wave_status: {summary.get('wave_status')}",
        f"- selected_track_count: {summary.get('selected_track_count')}",
        f"- selected_track_ids: {', '.join(summary.get('selected_track_ids') or [])}",
        f"- generation_started: {str(summary.get('generation_started') is True).lower()}",
        f"- listen_preview_count: {summary.get('listen_preview_count')}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        "",
        "## Semantic Batch",
        "",
    ]
    semantic = summary.get("semantic_batch") if isinstance(summary.get("semantic_batch"), dict) else {}
    for key in ("batch_status", "track_count", "candidate_count", "accepted_candidates", "blocked_hard_case"):
        lines.append(f"- {key}: {semantic.get(key)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist_summary(output_root: Path, summary: dict[str, Any]) -> None:
    write_json(output_root / SUMMARY_JSON_NAME, summary)
    write_summary_md(output_root / SUMMARY_MD_NAME, summary)


def run_wave(
    *,
    job_matrix_jsonl: Path,
    output_root: Path,
    existing_listen_roots: Sequence[Path],
    max_tracks: int,
    max_jobs_per_track: int = 4,
    execute_generation: bool = False,
    runtime_root: Path = Path("/app"),
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / PROGRESS_JSONL_NAME
    selected_tracks = select_next_track_ids(
        job_matrix_jsonl=job_matrix_jsonl,
        existing_listen_roots=existing_listen_roots,
        max_tracks=max_tracks,
    )
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_nonvocal_bounded_wave_start",
            "selected_track_ids": selected_tracks,
            "selected_track_count": len(selected_tracks),
            "execute_generation": execute_generation,
            "max_jobs_per_track": max_jobs_per_track,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring" if execute_generation else "dry_run_no_inference",
        },
    )
    base_summary: dict[str, Any] = {
        "wave_status": "dry_run" if not execute_generation else "blocked_no_selected_tracks",
        "selected_track_ids": selected_tracks,
        "selected_track_count": len(selected_tracks),
        "generation_scope": "selected_tracks_only",
        "semantic_batch_scope": "cumulative_existing_plus_new_listen_roots",
        "semantic_batch_track_filter": None,
        "existing_listen_root_count": len(existing_listen_roots),
        "generation_started": False,
        "listen_preview_count": 0,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "safety": dict(SAFETY_FLAGS),
        "semantic_batch": {},
    }
    if not selected_tracks or not execute_generation:
        base_summary["elapsed_sec"] = round(time.time() - started, 6)
        persist_summary(output_root, base_summary)
        append_jsonl(progress_path, {"event": "no_reference_nonvocal_bounded_wave_end", **base_summary})
        return base_summary

    execution_root = output_root / "job_execution"
    max_jobs = max_jobs_per_track * len(selected_tracks)
    append_jsonl(
        progress_path,
        {
            "event": "generation_start",
            "selected_track_ids": selected_tracks,
            "selected_track_count": len(selected_tracks),
            "max_jobs": max_jobs,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
        },
    )
    generation = execute_job_matrix(
        job_matrix_jsonl=job_matrix_jsonl,
        output_root=execution_root,
        max_jobs=max_jobs,
        stems=list(TARGET_STEMS),
        track_ids=selected_tracks,
        dry_run=False,
        runtime_root=runtime_root,
    )
    generation_summary = generation.get("summary", {})
    append_jsonl(
        progress_path,
        {
            "event": "generation_end",
            "execution_status_counts": generation_summary.get("execution_status_counts", {}),
            "completed_jobs": generation_summary.get("completed_jobs"),
            "failed_jobs": generation_summary.get("failed_jobs"),
        },
    )
    listen_root = output_root / "listen_preview"
    append_jsonl(progress_path, {"event": "listen_preview_start", "output_root": str(listen_root)})
    listen_manifests = build_listen_previews_from_execution_rows(
        rows=generation.get("rows", []),
        output_root=listen_root,
    )
    append_jsonl(
        progress_path,
        {
            "event": "listen_preview_end",
            "listen_preview_count": len(listen_manifests),
        },
    )
    semantic_roots = list(existing_listen_roots) + ([listen_root] if listen_manifests else [])
    append_jsonl(
        progress_path,
        {
            "event": "semantic_batch_start",
            "semantic_batch_scope": "cumulative_existing_plus_new_listen_roots",
            "semantic_batch_track_filter": None,
            "semantic_manifest_root_count": len(semantic_roots),
        },
    )
    semantic_summary = run_semantic_batch(
        listen_manifest_roots=semantic_roots,
        output_root=output_root / "semantic_batch",
    )
    append_jsonl(
        progress_path,
        {
            "event": "semantic_batch_end",
            "batch_status": semantic_summary.get("batch_status"),
            "track_count": semantic_summary.get("track_count"),
            "candidate_count": semantic_summary.get("candidate_count"),
            "blocked_hard_case": semantic_summary.get("blocked_hard_case"),
        },
    )
    summary = {
        **base_summary,
        "wave_status": "completed",
        "generation_started": True,
        "generation_summary": generation_summary,
        "listen_preview_count": len(listen_manifests),
        "listen_manifest_paths": [str(path) for path in listen_manifests],
        "semantic_manifest_root_count": len(semantic_roots),
        "semantic_batch": semantic_summary,
        "elapsed_sec": round(time.time() - started, 6),
    }
    persist_summary(output_root, summary)
    append_jsonl(
        progress_path,
        {
            "event": "no_reference_nonvocal_bounded_wave_end",
            "wave_status": summary["wave_status"],
            "selected_track_count": len(selected_tracks),
            "listen_preview_count": len(listen_manifests),
            "elapsed_sec": summary["elapsed_sec"],
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_no_reference_nonvocal_bounded_wave.py",
        entrypoint_path="/app/tools/run_no_reference_nonvocal_bounded_wave.py",
    )
    parser = argparse.ArgumentParser(description="Run a bounded non-vocal corpus wave: generation, listen-preview, semantic batch.")
    parser.add_argument("--job-matrix-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--existing-listen-root", type=Path, action="append", default=[])
    parser.add_argument("--max-tracks", type=int, default=1)
    parser.add_argument("--max-jobs-per-track", type=int, default=4)
    parser.add_argument("--execute-generation", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    args = parser.parse_args(argv)

    summary = run_wave(
        job_matrix_jsonl=args.job_matrix_jsonl,
        output_root=args.output_root,
        existing_listen_roots=args.existing_listen_root,
        max_tracks=args.max_tracks,
        max_jobs_per_track=args.max_jobs_per_track,
        execute_generation=args.execute_generation,
        runtime_root=args.runtime_root,
    )
    print(
        json.dumps(
            {
                "wave_status": summary["wave_status"],
                "selected_track_count": summary["selected_track_count"],
                "selected_track_ids": summary["selected_track_ids"],
                "generation_started": summary["generation_started"],
                "listen_preview_count": summary["listen_preview_count"],
                "release_ready": summary["release_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
