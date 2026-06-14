from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.build_no_reference_ml_dataset import stable_hash  # noqa: E402


TEXT_ARTIFACTS = (
    ("run_manifest", "run_manifest.json", True),
    ("runtime_summary", "runtime_summary.md", True),
    ("candidate_pool", "candidate_pool.jsonl", True),
    ("candidate_wave_summary", "candidates/candidate_wave_summary.json", True),
    ("candidate_wave_manifest", "candidates/candidate_wave_manifest.json", False),
    ("candidate_wave_jobs", "candidates/candidate_wave_jobs.jsonl", True),
    ("candidate_wave_failures", "candidates/candidate_wave_failures.jsonl", True),
    ("hard_case_candidate_wave_summary", "candidates_minus_hard_case/candidate_wave_summary.json", False),
    ("hard_case_candidate_wave_manifest", "candidates_minus_hard_case/candidate_wave_manifest.json", False),
    ("hard_case_candidate_wave_jobs", "candidates_minus_hard_case/candidate_wave_jobs.jsonl", False),
    ("hard_case_candidate_wave_failures", "candidates_minus_hard_case/candidate_wave_failures.jsonl", False),
    ("vocal_windowed_judge", "vocal_windowed_judge.jsonl", True),
    ("minus_windowed_judge", "minus_windowed_judge.jsonl", True),
    ("learned_critic_scores", "learned_critic_scores.jsonl", True),
    ("learned_critic_disagreements", "learned_critic_disagreements.jsonl", False),
    ("optimizer_mode_specific_result", "optimizer_mode_specific_result.jsonl", False),
    ("selected_candidates", "selected_candidates.jsonl", True),
    ("pair_compatibility", "pair_compatibility.jsonl", True),
    ("minus_hard_case_router", "minus_hard_case_router.jsonl", False),
    ("preview_manifest", "preview_manifest.json", True),
    ("rendered_preview_audio", "rendered_preview_audio.jsonl", True),
    ("preview_render_contract", "preview_render_contract.json", True),
    ("preview_render_summary", "preview_render_summary.md", False),
    ("post_render_judge", "post_render_judge.jsonl", True),
    ("learning_outcome", "learning_outcome.jsonl", True),
)
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
SAFETY_KEYS = (
    "reference_required",
    "human_csv_required",
    "training_started",
    "separator_training_started",
    "model_downloaded",
    "model_registry_edited",
    "thresholds_relaxed",
    "cleanup_routes_added",
    "release_ready",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_artifact_row(run_dir: Path, artifact_role: str, relative_path: str, required: bool) -> dict[str, Any]:
    path = run_dir / relative_path
    row: dict[str, Any] = {
        "event_type": "artifact_index",
        "artifact_role": artifact_role,
        "run_root": str(run_dir),
        "relative_path": relative_path,
        "source_path": str(path),
        "exists": path.exists(),
        "required": bool(required),
        "copy_status": "metadata_only",
        "release_ready": False,
    }
    if path.exists():
        row.update({"size_bytes": path.stat().st_size, "sha256": file_digest(path)})
        if path.suffix == ".jsonl":
            row["row_count"] = len(read_jsonl(path))
        elif path.suffix == ".json":
            payload = read_json(path)
            row["json_keys"] = sorted(payload.keys())
    return row


def audio_references_from_rows(rows: Iterable[dict[str, Any]], *, run_dir: Path, source_event: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or row.get("candidate_config_id") or "")
        for key, value in row.items():
            if not key.endswith("_path") and key not in {"minus_path", "vocal_path", "instrumental_path", "input_audio"}:
                continue
            if not value:
                continue
            path = Path(str(value))
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            marker = str(path)
            if marker in seen:
                continue
            seen.add(marker)
            ref = {
                "event_type": "artifact_index",
                "artifact_role": "audio_reference",
                "source_event": source_event,
                "run_root": str(run_dir),
                "candidate_id": candidate_id,
                "field": key,
                "source_path": marker,
                "exists": path.exists(),
                "copy_status": "metadata_only",
                "release_ready": False,
            }
            if path.exists() and path.is_file():
                ref.update({"size_bytes": path.stat().st_size, "sha256": file_digest(path)})
            references.append(ref)
    return references


def is_accept(row: dict[str, Any]) -> bool:
    values = {
        str(row.get("decision") or ""),
        str(row.get("script_decision") or ""),
        str(row.get("final_guarded_decision") or ""),
        str(row.get("minus_decision") or ""),
        str(row.get("vocal_decision") or ""),
    }
    return bool(values & {"accept_candidate", "accepted", "usable_candidate", "compatible", "preview_ready"})


def blocker_counter(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        values: list[Any] = []
        for key in ("minus_blockers", "blockers", "script_blockers", "failure_reasons", "top_failure_reasons"):
            value = row.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif isinstance(value, dict):
                values.extend(value.keys())
            elif value:
                values.append(value)
        for value in values:
            text = str(value)
            counts[text] = counts.get(text, 0) + 1
    return counts


def release_ready_source_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("release_ready") is True or row.get("can_mark_release_ready") is True)


def bool_value(*values: Any) -> bool:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
    return False


def int_value(*values: Any) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def validated_rendered_preview_count(rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        render_status = str(row.get("render_status") or row.get("status") or "rendered")
        validation_status = str(row.get("validation_status") or row.get("preview_validation_status") or "validated")
        if render_status in {"rendered", "success", "ok"} and validation_status in {"validated", "accepted", "ok"}:
            count += 1
    return count


def expanded_runtime_run_roots(run_roots: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[str] = set()
    for root in run_roots:
        run_root = Path(root)
        candidates = [run_root] if (run_root / "run_manifest.json").exists() else []
        if not candidates:
            candidates = sorted(path.parent for path in (run_root / "runs").glob("*/*/run_manifest.json"))
        if not candidates:
            candidates = [run_root]
        for candidate in candidates:
            marker = str(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            expanded.append(candidate)
    return expanded


def run_roots_from_jsonl(path: Path) -> list[Path]:
    roots: list[Path] = []
    for row in read_jsonl(path):
        if row.get("preview_contract_complete") is False:
            continue
        root = row.get("run_root") or row.get("run_dir") or row.get("path")
        if root:
            roots.append(Path(str(root)))
    return roots


def run_summary_event(run_dir: Path, artifact_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    pool_rows = read_jsonl(run_dir / "candidate_pool.jsonl")
    vocal_rows = read_jsonl(run_dir / "vocal_windowed_judge.jsonl")
    minus_rows = read_jsonl(run_dir / "minus_windowed_judge.jsonl")
    critic_rows = read_jsonl(run_dir / "learned_critic_scores.jsonl")
    selected_rows = read_jsonl(run_dir / "selected_candidates.jsonl")
    pair_rows = read_jsonl(run_dir / "pair_compatibility.jsonl")
    post_rows = read_jsonl(run_dir / "post_render_judge.jsonl")
    preview_manifest = read_json(run_dir / "preview_manifest.json")
    rendered_preview_rows = read_jsonl(run_dir / "rendered_preview_audio.jsonl")
    preview_contract = read_json(run_dir / "preview_render_contract.json")
    learning_rows = read_jsonl(run_dir / "learning_outcome.jsonl")
    candidate_wave_summary = read_json(run_dir / "candidates" / "candidate_wave_summary.json")
    hard_wave_summary = read_json(run_dir / "candidates_minus_hard_case" / "candidate_wave_summary.json")
    missing_required = [
        row["relative_path"]
        for row in artifact_rows
        if row.get("required") is True and row.get("exists") is not True
    ]
    blockers = blocker_counter(minus_rows)
    replay_complete = not missing_required
    preview_contract_missing = not (run_dir / "preview_render_contract.json").exists()
    preview_status = first_text(
        preview_contract.get("preview_status"),
        preview_manifest.get("preview_status"),
        preview_manifest.get("runtime_status"),
        preview_manifest.get("presentation_status"),
        manifest.get("output_status"),
    )
    validation_status = first_text(preview_contract.get("validation_status"), preview_manifest.get("validation_status"))
    accepted_pair_count = sum(1 for row in selected_rows if str(row.get("score_mode") or row.get("mode") or "") == "pair")
    pair_compatible_count = sum(1 for row in pair_rows if is_accept(row))
    accepted_pair_ready = bool_value(preview_contract.get("accepted_pair_ready"), preview_manifest.get("accepted_pair_ready")) or accepted_pair_count > 0 or pair_compatible_count > 0
    manifest_preview_ready = bool_value(preview_contract.get("manifest_preview_ready"), preview_manifest.get("manifest_preview_ready")) or preview_status in {
        "manifest_preview_ready",
        "audio_preview_ready",
        "preview_ready",
        "accepted_for_listen_pack",
    }
    rendered_preview_audio_count = validated_rendered_preview_count(rendered_preview_rows)
    if not rendered_preview_rows:
        rendered_preview_audio_count = max(
            rendered_preview_audio_count,
            int_value(preview_contract.get("rendered_preview_audio_count"), preview_manifest.get("rendered_preview_audio_count")),
        )
    requested_audio_preview_ready = bool_value(preview_contract.get("audio_preview_ready"), preview_manifest.get("audio_preview_ready")) or preview_status == "audio_preview_ready"
    audio_preview_ready = (
        requested_audio_preview_ready
        and not preview_contract_missing
        and rendered_preview_audio_count > 0
        and validation_status in {"validated", "accepted", "ok"}
    )
    preview_render_attempted = bool_value(preview_contract.get("preview_render_attempted"), preview_manifest.get("preview_render_attempted")) or bool(rendered_preview_rows)
    preview_render_skipped = bool_value(preview_contract.get("preview_render_skipped"), preview_manifest.get("preview_render_skipped")) or preview_status == "preview_render_skipped"
    preview_render_failed = bool_value(preview_contract.get("preview_render_failed"), preview_manifest.get("preview_render_failed")) or preview_status == "preview_render_failed"
    preview_validation_failed = (
        bool_value(preview_contract.get("preview_validation_failed"), preview_manifest.get("preview_validation_failed"))
        or preview_status == "preview_validation_failed"
        or validation_status in {"failed", "invalid", "blocked"}
        or (requested_audio_preview_ready and not audio_preview_ready)
    )
    preview_failure_reason = first_text(preview_contract.get("preview_failure_reason"), preview_manifest.get("preview_failure_reason"), preview_contract.get("failure_reason"))
    preview_skip_reason = first_text(preview_contract.get("preview_skip_reason"), preview_manifest.get("preview_skip_reason"), preview_contract.get("skip_reason"))
    failed_safety = (
        str(manifest.get("output_status") or "") == "failed_safety_contract"
        or "failed_safety_contract" in set(manifest.get("safety_blockers") or [])
    )
    jobs_attempted = int(candidate_wave_summary.get("jobs_attempted") or 0) + int(hard_wave_summary.get("jobs_attempted") or 0)
    estimated_runtime = int(candidate_wave_summary.get("estimated_runtime_sec") or candidate_wave_summary.get("elapsed_sec") or 0)
    estimated_runtime += int(hard_wave_summary.get("estimated_runtime_sec") or hard_wave_summary.get("elapsed_sec") or 0)
    return {
        "event_type": "run_summary",
        "run_id": stable_hash({"run_root": str(run_dir)})[:16],
        "run_root": str(run_dir),
        "track_id": str(manifest.get("track_id") or Path(str(manifest.get("input_audio") or run_dir.name)).stem),
        "input_audio": str(manifest.get("input_audio") or manifest.get("input_audio_path") or ""),
        "replay_complete": replay_complete,
        "missing_artifacts": missing_required,
        "output_status": str(manifest.get("output_status") or ""),
        "failure_reason": str(manifest.get("failure_reason") or ""),
        "accepted_pair_count": accepted_pair_count,
        "accepted_pair_ready": accepted_pair_ready,
        "manifest_preview_ready": manifest_preview_ready,
        "audio_preview_ready": audio_preview_ready,
        "accepted_pair_ready_count": 1 if accepted_pair_ready else 0,
        "manifest_preview_ready_count": 1 if manifest_preview_ready else 0,
        "audio_preview_ready_count": 1 if audio_preview_ready else 0,
        "rendered_preview_audio_count": rendered_preview_audio_count,
        "preview_ready_count": 1 if audio_preview_ready else 0,
        "preview_render_attempted": preview_render_attempted,
        "preview_render_skipped": preview_render_skipped,
        "preview_render_failed": preview_render_failed,
        "preview_validation_failed": preview_validation_failed,
        "preview_contract_missing": preview_contract_missing,
        "preview_render_failed_count": 1 if preview_render_failed else 0,
        "preview_render_skipped_count": 1 if preview_render_skipped else 0,
        "preview_validation_failed_count": 1 if preview_validation_failed else 0,
        "preview_contract_missing_count": 1 if preview_contract_missing else 0,
        "preview_status": preview_status,
        "preview_failure_reason": preview_failure_reason,
        "preview_skip_reason": preview_skip_reason,
        "can_publish_as_result": False,
        "can_mark_release_ready": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "final_safe_candidate_count": len(selected_rows),
        "pair_compatible_count": pair_compatible_count,
        "minus_accepted_count": sum(1 for row in minus_rows + critic_rows if str(row.get("score_mode") or "minus") == "minus" and is_accept(row)),
        "vocal_accepted_count": sum(1 for row in vocal_rows + critic_rows if str(row.get("score_mode") or "vocal") == "vocal" and is_accept(row)),
        "severe_vocal_bleed_in_minus_count": int(blockers.get("severe_vocal_bleed_in_minus", 0)),
        "severe_phase_collapse_count": int(blockers.get("severe_phase_collapse", 0)),
        "failed_safety_contract_count": 1 if failed_safety else 0,
        "release_ready_count": release_ready_source_count(
            [manifest, preview_manifest, preview_contract] + pool_rows + critic_rows + selected_rows + pair_rows + post_rows + rendered_preview_rows + learning_rows
        ),
        "estimated_gpu_jobs": jobs_attempted,
        "estimated_runtime_sec": estimated_runtime,
        "duplicate_jobs_avoided": int(candidate_wave_summary.get("duplicate_jobs_avoided") or 0) + int(hard_wave_summary.get("duplicate_jobs_avoided") or 0),
        "candidate_count": len(pool_rows),
        "complexity_score": len(pool_rows) + jobs_attempted + (1 if hard_wave_summary else 0),
        "failed_run_count": 0 if str(manifest.get("output_status") or "") not in {"failed_safety_contract"} else 1,
        "timeout_count": int(candidate_wave_summary.get("jobs_failed_timeout") or 0) + int(hard_wave_summary.get("jobs_failed_timeout") or 0),
        "incomplete_replay_count": 0 if replay_complete else 1,
        **{key: False if key == "release_ready" else manifest.get(key, False) for key in SAFETY_KEYS},
    }


def event_rows_for_run(run_dir: Path, artifact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = run_summary_event(run_dir, artifact_rows)
    rows.append(summary)
    if not summary["replay_complete"]:
        rows.append(
            {
                "event_type": "replay_incomplete",
                "run_root": str(run_dir),
                "track_id": summary["track_id"],
                "missing_artifacts": summary["missing_artifacts"],
                "release_ready": False,
            }
        )
    for artifact_role, relative_path, _required in TEXT_ARTIFACTS:
        path = run_dir / relative_path
        if not path.exists() or path.suffix != ".jsonl":
            continue
        for source_row in read_jsonl(path):
            rows.append(
                {
                    "event_type": artifact_role,
                    "run_root": str(run_dir),
                    "track_id": summary["track_id"],
                    "payload": {**source_row, "source_release_ready": source_row.get("release_ready"), "release_ready": False},
                    "release_ready": False,
                }
            )
    return rows


def job_manifest_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_name in ("candidates", "candidates_minus_hard_case"):
        for path in sorted((run_dir / root_name / "jobs").glob("*/job_manifest.json")):
            rows.append(text_artifact_row(run_dir, f"{root_name}_job_manifest", str(path.relative_to(run_dir)), False))
        for path in sorted((run_dir / root_name / "jobs").glob("*/job_progress.jsonl")):
            rows.append(text_artifact_row(run_dir, f"{root_name}_job_progress", str(path.relative_to(run_dir)), False))
    return rows


def build_replay_bundle(*, run_roots: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    expanded_run_roots = expanded_runtime_run_roots(run_roots)
    for run_root in expanded_run_roots:
        run_dir = Path(run_root)
        rows_for_artifacts = [text_artifact_row(run_dir, role, rel_path, required) for role, rel_path, required in TEXT_ARTIFACTS]
        rows_for_artifacts.extend(job_manifest_rows(run_dir))
        artifact_rows.extend(rows_for_artifacts)
        summary = run_summary_event(run_dir, rows_for_artifacts)
        run_summaries.append(summary)
        event_rows.extend(event_rows_for_run(run_dir, rows_for_artifacts))
        for rel_path in ("candidate_pool.jsonl", "rendered_preview_audio.jsonl", "candidates/candidate_wave_jobs.jsonl", "candidates/candidate_wave_failures.jsonl"):
            artifact_rows.extend(audio_references_from_rows(read_jsonl(run_dir / rel_path), run_dir=run_dir, source_event=rel_path))
        manifest = read_json(run_dir / "run_manifest.json")
        artifact_rows.extend(audio_references_from_rows([manifest], run_dir=run_dir, source_event="run_manifest"))
    manifest = {
        "schema_version": 1,
        "bundle_id": "replay_bundle_" + stable_hash([str(Path(root)) for root in run_roots])[:16],
        "run_roots": [str(Path(root)) for root in run_roots],
        "expanded_run_roots": [str(Path(root)) for root in expanded_run_roots],
        "run_count": len(run_summaries),
        "replay_complete_runs": sum(1 for row in run_summaries if row["replay_complete"] is True),
        "replay_incomplete_runs": sum(1 for row in run_summaries if row["replay_complete"] is not True),
        "release_ready_count": sum(int(row.get("release_ready_count") or 0) for row in run_summaries),
        "accepted_pair_ready_count": sum(int(row.get("accepted_pair_ready_count") or 0) for row in run_summaries),
        "manifest_preview_ready_count": sum(int(row.get("manifest_preview_ready_count") or 0) for row in run_summaries),
        "audio_preview_ready_count": sum(int(row.get("audio_preview_ready_count") or 0) for row in run_summaries),
        "rendered_preview_audio_count": sum(int(row.get("rendered_preview_audio_count") or 0) for row in run_summaries),
        "preview_render_failed_count": sum(int(row.get("preview_render_failed_count") or 0) for row in run_summaries),
        "preview_render_skipped_count": sum(int(row.get("preview_render_skipped_count") or 0) for row in run_summaries),
        "preview_validation_failed_count": sum(int(row.get("preview_validation_failed_count") or 0) for row in run_summaries),
        "preview_contract_missing_count": sum(int(row.get("preview_contract_missing_count") or 0) for row in run_summaries),
        "audio_files_copied": False,
        "reference_required": False,
        "human_csv_required": False,
        "training_started": False,
        "separator_training_started": False,
        "model_downloaded": False,
        "model_registry_edited": False,
        "thresholds_relaxed": False,
        "cleanup_routes_added": False,
        "melody_transcription_ran": False,
        "recommendation_index_updated": False,
        "release_ready": False,
    }
    write_json(output_dir / "replay_bundle_manifest.json", manifest)
    write_jsonl(output_dir / "replay_bundle_events.jsonl", event_rows)
    write_jsonl(output_dir / "replay_bundle_artifacts_index.jsonl", artifact_rows)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a metadata-only replay bundle from autonomous no-reference runtime runs.")
    parser.add_argument("--run-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--run-roots-from-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_roots = list(args.run_roots)
    if args.run_roots_from_jsonl is not None:
        run_roots.extend(run_roots_from_jsonl(args.run_roots_from_jsonl))
    if not run_roots:
        raise SystemExit("--run-roots or --run-roots-from-jsonl is required")
    manifest = build_replay_bundle(run_roots=run_roots, output_dir=args.output_dir)
    print(json.dumps({"replay_bundle_manifest": str(args.output_dir / "replay_bundle_manifest.json"), **manifest}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
