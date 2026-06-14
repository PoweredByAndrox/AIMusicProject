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

from tools.build_testing_stuff_music_corpus_job_matrix import registry_job, safe_slug  # noqa: E402
from tools.docker_runtime import require_docker_runtime  # noqa: E402
from tools.run_no_reference_nonvocal_bounded_wave import build_listen_previews_from_execution_rows  # noqa: E402
from tools.run_no_reference_stem_semantic_batch import run_batch as run_semantic_batch  # noqa: E402
from tools.run_testing_stuff_music_corpus_job_matrix import execute_job_matrix  # noqa: E402
from worker.model_registry import ModelRegistry, default_model_registry  # noqa: E402


SUMMARY_JSON_NAME = "no_reference_stem_hard_case_repair_wave_summary.json"
SUMMARY_MD_NAME = "no_reference_stem_hard_case_repair_wave_summary.md"
PROGRESS_JSONL_NAME = "no_reference_stem_hard_case_repair_wave_progress.jsonl"
REPAIR_JOBS_JSONL_NAME = "hard_case_repair_jobs.jsonl"

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


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object row: {path}")
        rows.append(payload)
    return rows


def source_ref_path(row: dict[str, Any]) -> Path | None:
    ref = row.get("source_listen_manifest")
    if isinstance(ref, dict) and str(ref.get("path") or ""):
        return Path(str(ref["path"]))
    return None


def stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def judge_decision(row: dict[str, Any]) -> str:
    return str(row.get("decision") or row.get("stem_semantic_decision") or "")


def judge_stem(row: dict[str, Any], listen_manifest: dict[str, Any] | None = None) -> str:
    listen_manifest = listen_manifest or {}
    return str(row.get("stem_type") or listen_manifest.get("stem_type") or "")


def evaluated_labels_from(listen_manifest: dict[str, Any], judge_row: dict[str, Any]) -> list[str]:
    labels: set[str] = set()
    for value in (
        judge_row.get("model_name"),
        listen_manifest.get("model_name"),
    ):
        text = str(value or "").strip()
        if text:
            labels.add(text)
    selected_audio_path = str(listen_manifest.get("selected_audio_path") or judge_row.get("selected_audio_path") or "")
    if selected_audio_path:
        labels.add(Path(selected_audio_path).parent.name)
    source_stem_manifest = listen_manifest.get("source_stem_manifest")
    if isinstance(source_stem_manifest, dict) and str(source_stem_manifest.get("path") or ""):
        labels.add(Path(str(source_stem_manifest["path"])).parent.name)
    return sorted(labels)


def listen_manifest_for_row(row: dict[str, Any], cache: dict[Path, dict[str, Any]]) -> dict[str, Any] | None:
    listen_path = source_ref_path(row)
    if listen_path is None or not listen_path.exists():
        return None
    if listen_path not in cache:
        cache[listen_path] = read_json(listen_path)
    return cache[listen_path]


def evaluated_labels_for_rows(
    rows: Sequence[dict[str, Any]],
    *,
    target_stem: str,
    manifest_cache: dict[Path, dict[str, Any]],
) -> list[str]:
    labels: list[str] = []
    for row in rows:
        listen_manifest = listen_manifest_for_row(row, manifest_cache) or {}
        if judge_stem(row, listen_manifest) != target_stem:
            continue
        labels.extend(evaluated_labels_from(listen_manifest, row))
    return stable_unique(labels)


def failed_labels_for_rows(rows: Sequence[dict[str, Any]], *, target_stem: str) -> list[str]:
    labels: list[str] = []
    for row in rows:
        if judge_decision(row) == "accept_candidate":
            continue
        if judge_stem(row) != target_stem:
            continue
        labels.extend(str(label) for label in row.get("stem_semantic_blockers") or [] if str(label))
    return stable_unique(labels)


def blocked_judge_rows(semantic_batch_summary_json: Path) -> list[tuple[dict[str, Any], dict[str, Any], list[str], list[str]]]:
    summary = read_json(semantic_batch_summary_json)
    rows: list[tuple[dict[str, Any], dict[str, Any], list[str], list[str]]] = []
    for track in summary.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        judge_jsonl = str(track.get("judge_jsonl") or "")
        if not judge_jsonl:
            continue
        track_rows = iter_jsonl(Path(judge_jsonl))
        manifest_cache: dict[Path, dict[str, Any]] = {}
        for row in track_rows:
            if judge_decision(row) == "accept_candidate":
                continue
            listen_manifest = listen_manifest_for_row(row, manifest_cache)
            if listen_manifest is None:
                continue
            target_stem = judge_stem(row, listen_manifest)
            evaluated_labels = evaluated_labels_for_rows(
                track_rows,
                target_stem=target_stem,
                manifest_cache=manifest_cache,
            )
            failed_labels = failed_labels_for_rows(track_rows, target_stem=target_stem)
            rows.append((row, listen_manifest, evaluated_labels, failed_labels))
    return rows


def repair_suggestion_sort_key(suggestion: dict[str, Any], target_stem: str) -> tuple[int, int, str]:
    target_stems = {str(stem) for stem in suggestion.get("target_stems") or []}
    exact_target_rank = 0 if target_stems == {target_stem} else 1
    family_rank = {"hybrid": 0, "demucs": 1, "dsp": 2}.get(str(suggestion.get("family") or ""), 3)
    return (exact_target_rank, family_rank, str(suggestion.get("model_id") or ""))


def next_repair_suggestion(
    *,
    registry: ModelRegistry,
    target_stem: str,
    failed_labels: Sequence[str],
    evaluated_labels: Sequence[str],
) -> dict[str, Any] | None:
    suggestions = [
        suggestion.to_dict()
        for suggestion in registry.suggest_next_candidates(
            stem_names=[target_stem],
            failed_labels=failed_labels,
            evaluated_labels=evaluated_labels,
            include_unavailable=False,
            limit=len(registry.entries),
        )
    ]
    if not suggestions:
        suggestions = [
            suggestion.to_dict()
            for suggestion in registry.suggest_next_candidates(
                stem_names=[target_stem],
                failed_labels=[],
                evaluated_labels=evaluated_labels,
                include_unavailable=False,
                limit=len(registry.entries),
            )
        ]
    suggestions.sort(key=lambda suggestion: repair_suggestion_sort_key(suggestion, target_stem))
    return suggestions[0] if suggestions else None


def plan_hard_case_repair_jobs(
    *,
    semantic_batch_summary_json: Path,
    output_root: Path,
    max_repair_jobs: int,
    registry: ModelRegistry | None = None,
    runtime_root: Path = Path("/app"),
    device: str = "cuda",
) -> list[dict[str, Any]]:
    if max_repair_jobs <= 0:
        raise ValueError("--max-repair-jobs must be positive")
    registry = registry or default_model_registry(runtime_root=runtime_root)
    jobs: list[dict[str, Any]] = []
    seen_track_stems: set[tuple[str, str]] = set()
    for judge_row, listen_manifest, evaluated_labels, failed_labels in blocked_judge_rows(semantic_batch_summary_json):
        stem = str(judge_row.get("stem_type") or listen_manifest.get("stem_type") or "")
        track_id = str(judge_row.get("track_id") or listen_manifest.get("track_id") or "")
        source_audio_path = str(listen_manifest.get("source_audio_path") or "")
        if not stem or not track_id or not source_audio_path:
            continue
        track_stem_key = (track_id, stem)
        if track_stem_key in seen_track_stems:
            continue
        seen_track_stems.add(track_stem_key)
        suggestion = next_repair_suggestion(
            registry=registry,
            target_stem=stem,
            failed_labels=failed_labels,
            evaluated_labels=evaluated_labels,
        )
        if suggestion is None:
            continue
        corpus_row = {
            "index": 0,
            "track_id": track_id,
            "relative_path": track_id,
            "source_audio_path": source_audio_path,
        }
        job = registry_job(
            corpus_row=corpus_row,
            target_stem=stem,
            suggestion=suggestion,
            output_root=output_root / "repair_candidates",
            device=device,
        )
        job.update(
            {
                "phase": "hard_case_repair_candidate_generation",
                "repair_reason": "semantic_hard_case_blocked",
                "repair_failed_labels": failed_labels,
                "repair_evaluated_labels": evaluated_labels,
                "source_semantic_batch_summary": str(semantic_batch_summary_json),
                "source_listen_manifest": str(source_ref_path(judge_row) or ""),
                "repair_scope": "blocked_stem_only",
                "release_ready": False,
                "can_mark_release_ready": False,
                "can_publish_as_release": False,
            }
        )
        jobs.append(job)
        if len(jobs) >= max_repair_jobs:
            break
    return jobs


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# No-Reference Stem Hard-Case Repair Wave",
        "",
        f"- repair_status: {summary.get('repair_status')}",
        f"- planned_repair_jobs: {summary.get('planned_repair_jobs')}",
        f"- generation_started: {str(summary.get('generation_started') is True).lower()}",
        f"- listen_preview_count: {summary.get('listen_preview_count')}",
        f"- release_ready: {str(summary.get('release_ready') is True).lower()}",
        "",
        "## Semantic Batch After Repair",
        "",
    ]
    semantic = summary.get("semantic_batch_after_repair") if isinstance(summary.get("semantic_batch_after_repair"), dict) else {}
    for key in ("batch_status", "track_count", "candidate_count", "accepted_candidates", "blocked_hard_case"):
        lines.append(f"- {key}: {semantic.get(key)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist_summary(output_root: Path, summary: dict[str, Any]) -> None:
    write_json(output_root / SUMMARY_JSON_NAME, summary)
    write_summary_md(output_root / SUMMARY_MD_NAME, summary)


def run_repair_wave(
    *,
    semantic_batch_summary_json: Path,
    existing_listen_roots: Sequence[Path],
    output_root: Path,
    max_repair_jobs: int = 1,
    execute_generation: bool = False,
    runtime_root: Path = Path("/app"),
    device: str = "cuda",
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / PROGRESS_JSONL_NAME
    append_jsonl(
        progress_path,
        {
            "event": "stem_hard_case_repair_wave_start",
            "execute_generation": execute_generation,
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring" if execute_generation else "dry_run_no_inference",
            "max_repair_jobs": max_repair_jobs,
            "semantic_batch_summary_json": str(semantic_batch_summary_json),
        },
    )
    jobs = plan_hard_case_repair_jobs(
        semantic_batch_summary_json=semantic_batch_summary_json,
        output_root=output_root,
        max_repair_jobs=max_repair_jobs,
        runtime_root=runtime_root,
        device=device,
    )
    jobs_jsonl = output_root / REPAIR_JOBS_JSONL_NAME
    write_jsonl(jobs_jsonl, jobs)
    base_summary: dict[str, Any] = {
        "repair_status": "dry_run" if not execute_generation else "blocked_no_repair_jobs",
        "planned_repair_jobs": len(jobs),
        "repair_jobs_jsonl": str(jobs_jsonl),
        "generation_started": False,
        "listen_preview_count": 0,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "safety": dict(SAFETY_FLAGS),
        "semantic_batch_after_repair": {},
        "elapsed_sec": round(time.time() - started, 6),
    }
    if not jobs or not execute_generation:
        persist_summary(output_root, base_summary)
        append_jsonl(progress_path, {"event": "stem_hard_case_repair_wave_end", **base_summary})
        return base_summary

    append_jsonl(
        progress_path,
        {
            "event": "repair_generation_start",
            "planned_repair_jobs": len(jobs),
            "expected_acceleration_mode": "gpu_inference_then_cpu_dsp_scoring",
        },
    )
    generation = execute_job_matrix(
        job_matrix_jsonl=jobs_jsonl,
        output_root=output_root / "job_execution",
        max_jobs=len(jobs),
        dry_run=False,
        runtime_root=runtime_root,
    )
    generation_summary = generation.get("summary", {})
    append_jsonl(
        progress_path,
        {
            "event": "repair_generation_end",
            "completed_jobs": generation_summary.get("completed_jobs"),
            "failed_jobs": generation_summary.get("failed_jobs"),
        },
    )
    listen_root = output_root / "listen_preview"
    append_jsonl(progress_path, {"event": "repair_listen_preview_start", "output_root": str(listen_root)})
    listen_manifests = build_listen_previews_from_execution_rows(
        rows=generation.get("rows", []),
        output_root=listen_root,
    )
    append_jsonl(progress_path, {"event": "repair_listen_preview_end", "listen_preview_count": len(listen_manifests)})
    semantic_roots = list(existing_listen_roots) + ([listen_root] if listen_manifests else [])
    append_jsonl(
        progress_path,
        {
            "event": "semantic_batch_after_repair_start",
            "semantic_manifest_root_count": len(semantic_roots),
        },
    )
    semantic_summary = run_semantic_batch(
        listen_manifest_roots=semantic_roots,
        output_root=output_root / "semantic_batch_after_repair",
    )
    append_jsonl(
        progress_path,
        {
            "event": "semantic_batch_after_repair_end",
            "batch_status": semantic_summary.get("batch_status"),
            "track_count": semantic_summary.get("track_count"),
            "candidate_count": semantic_summary.get("candidate_count"),
            "blocked_hard_case": semantic_summary.get("blocked_hard_case"),
        },
    )
    summary = {
        **base_summary,
        "repair_status": "completed",
        "generation_started": True,
        "generation_summary": generation_summary,
        "listen_preview_count": len(listen_manifests),
        "listen_manifest_paths": [str(path) for path in listen_manifests],
        "semantic_manifest_root_count": len(semantic_roots),
        "semantic_batch_after_repair": semantic_summary,
        "elapsed_sec": round(time.time() - started, 6),
    }
    persist_summary(output_root, summary)
    append_jsonl(
        progress_path,
        {
            "event": "stem_hard_case_repair_wave_end",
            "repair_status": summary["repair_status"],
            "planned_repair_jobs": len(jobs),
            "listen_preview_count": len(listen_manifests),
            "elapsed_sec": summary["elapsed_sec"],
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    require_docker_runtime(
        tool_name="run_no_reference_stem_hard_case_repair_wave.py",
        entrypoint_path="/app/tools/run_no_reference_stem_hard_case_repair_wave.py",
    )
    parser = argparse.ArgumentParser(description="Run a bounded stem hard-case repair wave from semantic blockers.")
    parser.add_argument("--semantic-batch-summary-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--existing-listen-root", type=Path, action="append", default=[])
    parser.add_argument("--max-repair-jobs", type=int, default=1)
    parser.add_argument("--execute-generation", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path("/app"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    summary = run_repair_wave(
        semantic_batch_summary_json=args.semantic_batch_summary_json,
        existing_listen_roots=args.existing_listen_root,
        output_root=args.output_root,
        max_repair_jobs=args.max_repair_jobs,
        execute_generation=bool(args.execute_generation),
        runtime_root=args.runtime_root,
        device=str(args.device),
    )
    print(
        json.dumps(
            {
                "repair_status": summary["repair_status"],
                "planned_repair_jobs": summary["planned_repair_jobs"],
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
